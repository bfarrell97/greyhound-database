"""
Enhanced Elo Model - LITE VERSION (Fast Training)
==================================================
Same as train_elo_gb.py but with:
- Smaller parameter grid (12 combos instead of 48)
- Subsampled training data (300k instead of 800k)
- More verbose output
Expected time: 5-10 minutes
"""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def get_k(track):
    if track in METRO_TRACKS: return 40
    elif track in PROVINCIAL_TRACKS: return 32
    return 24

def train_elo_model():
    print("="*70)
    print("ENHANCED ELO MODEL - LITE VERSION (Fast)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/6] Loading race data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           r.Distance, rm.MeetingDate, t.TrackName, ge.Margin
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[2/6] Building Elo ratings and features...")
    
    elo_ratings = defaultdict(lambda: 1500)
    race_count = defaultdict(int)
    win_count = defaultdict(int)
    last_race_date = {}
    
    feature_rows = []
    races_processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        k = get_k(track)
        
        race_elo = {r['GreyhoundID']: elo_ratings[r['GreyhoundID']] for _, r in race_df.iterrows()}
        elo_sorted = sorted(race_elo.items(), key=lambda x: x[1], reverse=True)
        elo_leader = elo_sorted[0][0]
        elo_2nd = elo_sorted[1][0] if len(elo_sorted) > 1 else elo_leader
        
        total_exp = sum(np.exp(r / 400) for r in race_elo.values())
        expected = {d: np.exp(race_elo[d] / 400) / total_exp for d in race_elo}
        
        for idx, (_, r) in enumerate(race_df.iterrows()):
            dog_id = r['GreyhoundID']
            
            if dog_id in last_race_date:
                days_since = (race_date - last_race_date[dog_id]).days
            else:
                days_since = 60
            days_since = min(days_since, 90)
            
            elo_rank = [i+1 for i, (d, _) in enumerate(elo_sorted) if d == dog_id][0]
            gap_to_leader = race_elo[elo_leader] - race_elo[dog_id]
            gap_to_2nd = race_elo[elo_2nd] - race_elo[dog_id] if dog_id != elo_leader else 0
            
            total_races = race_count[dog_id]
            total_wins = win_count[dog_id]
            win_rate = total_wins / total_races if total_races > 0 else 0
            
            feature_rows.append({
                'RaceID': race_id,
                'GreyhoundID': dog_id,
                'Won': r['Won'],
                'BSP': r['BSP'],
                'Date': race_date,
                'Elo': race_elo[dog_id],
                'EloRank': elo_rank,
                'EloExpected': expected[dog_id],
                'EloGapToLeader': gap_to_leader,
                'EloGapTo2nd': gap_to_2nd,
                'IsEloLeader': 1 if dog_id == elo_leader else 0,
                'DaysSince': days_since,
                'RaceCount': min(total_races, 50),
                'WinRate': win_rate,
                'WinCount': min(total_wins, 20),
                'Box': r['Box'],
                'Distance': distance,
                'Tier': tier,
                'FieldSize': len(race_df)
            })
        
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            actual = 1.0 if r['Won'] else 0.0
            margin_bonus = min(r['Margin'] / 10, 0.5) if actual == 1.0 and r['Margin'] > 0 else 0
            elo_ratings[dog_id] += k * (1 + margin_bonus) * (actual - expected[dog_id])
            race_count[dog_id] += 1
            if actual == 1.0:
                win_count[dog_id] += 1
            last_race_date[dog_id] = race_date
        
        races_processed += 1
        if races_processed % 50000 == 0:
            print(f"  {races_processed:,} races...")
    
    print(f"  Total: {races_processed:,} races")
    
    print("\n[3/6] Preparing train/test split...")
    
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    feat_df = feat_df.dropna(subset=['BSP'])
    
    train_df = feat_df[feat_df['Date'] < datetime(2024, 1, 1)]
    test_df = feat_df[feat_df['Date'] >= datetime(2024, 1, 1)]
    
    # SUBSAMPLE for speed
    if len(train_df) > 300000:
        print(f"  Subsampling from {len(train_df):,} to 300,000 for speed...")
        train_df = train_df.sample(n=300000, random_state=42)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    feature_cols = ['Elo', 'EloRank', 'EloExpected', 'EloGapToLeader', 'EloGapTo2nd',
                    'IsEloLeader', 'DaysSince', 'RaceCount', 'WinRate', 'WinCount',
                    'Box', 'Distance', 'Tier', 'FieldSize']
    
    X_train = train_df[feature_cols]
    y_train = train_df['Won']
    X_test = test_df[feature_cols]
    y_test = test_df['Won']
    
    print("\n[4/6] Training with LITE GridSearchCV...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LITE parameter grid - only 12 combinations
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'min_samples_split': [100],
        'subsample': [0.8, 1.0]
    }
    
    print(f"Grid: {param_grid}")
    print("Running 12 fits (3-fold CV)...")
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    print("\n[5/6] Evaluating on test set...")
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print(f"\nTest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nML Top Pick (highest prob per race):")
    print(f"  Total races: {len(race_leaders):,}")
    print(f"  Wins: {race_leaders['Won'].sum():,}")
    print(f"  Strike Rate: {race_leaders['Won'].mean()*100:.1f}%")
    
    print("\n[6/6] Feature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS (2024-2025)")
    print("="*70)
    
    def backtest(df, label):
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        valid_bsp = df.dropna(subset=['BSP'])
        returns = valid_bsp[valid_bsp['Won'] == 1]['BSP'].sum()
        profit = returns - len(valid_bsp)
        roi = profit / len(valid_bsp) * 100 if len(valid_bsp) > 0 else 0
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- ALL TRACKS ---")
    backtest(race_leaders, "All picks")
    backtest(race_leaders[(race_leaders['BSP'] >= 2) & (race_leaders['BSP'] <= 10)], "$2-$10")
    backtest(race_leaders[(race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)], "$3-$8")
    
    print("\n--- BY TIER ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        t = race_leaders[race_leaders['Tier'] == tier]
        if len(t) > 100:
            t_filt = t[(t['BSP'] >= 2) & (t['BSP'] <= 10)]
            backtest(t_filt, f"{name} $2-$10")
    
    print("\n" + "="*70)
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_params': grid_search.best_params_
    }
    with open('models/elo_gb_model_lite.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved to models/elo_gb_model_lite.pkl")
    print("="*70)

if __name__ == "__main__":
    train_elo_model()
