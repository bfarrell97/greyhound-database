"""
Enhanced Pace Model - LITE VERSION (Fast Training)
===================================================
Same concept as train_pace_gb.py but with:
- Smaller parameter grid (4 combos)
- Subsampled training data (300k)
- More verbose output
Expected time: 5-10 minutes
"""
import sqlite3
import pandas as pd
import numpy as np
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

def train_pace_model():
    print("="*70)
    print("ENHANCED PACE MODEL - LITE VERSION (Fast)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/7] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    print(f"Loaded {len(bench_lookup):,} benchmarks")
    
    print("\n[2/7] Loading race data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2019-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    df['FinishTime'] = pd.to_numeric(df['FinishTime'], errors='coerce')
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    
    print(f"Loaded {len(df):,} entries")
    print(f"With valid times: {df['NormTime'].notna().sum():,}")
    
    print("\n[3/7] Building pace features...")
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
    dog_history = {}
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            hist = dog_history.get(dog_id, [])
            
            if len(hist) >= 3:
                recent_times = [h[1] for h in hist[-5:] if h[1] is not None and not np.isnan(h[1])]
                recent_splits = [h[2] for h in hist[-5:] if h[2] is not None and not np.isnan(h[2])]
                recent_pos = [h[3] for h in hist[-5:] if h[3] is not None]
                
                if len(recent_times) >= 3:
                    time_lag1 = recent_times[-1]
                    time_lag2 = recent_times[-2] if len(recent_times) >= 2 else time_lag1
                    time_lag3 = recent_times[-3] if len(recent_times) >= 3 else time_lag2
                    time_avg3 = np.mean(recent_times[-3:])
                    time_avg5 = np.mean(recent_times[-5:]) if len(recent_times) >= 5 else time_avg3
                    time_std = np.std(recent_times[-5:]) if len(recent_times) >= 3 else 0
                    time_trend = time_lag1 - time_lag3
                    time_best = min(recent_times)
                    
                    split_avg = np.mean(recent_splits) if recent_splits else 0
                    split_lag1 = recent_splits[-1] if recent_splits else 0
                    
                    pos_avg = np.mean(recent_pos) if recent_pos else 4
                    pos_lag1 = recent_pos[-1] if recent_pos else 4
                    win_rate = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                    place_rate = sum(1 for p in recent_pos if p <= 3) / len(recent_pos) if recent_pos else 0
                    
                    last_date = hist[-1][0]
                    days_since = min((race_date - last_date).days, 90)
                    race_count = min(len(hist), 50)
                    
                    feature_rows.append({
                        'RaceID': race_id,
                        'GreyhoundID': dog_id,
                        'Won': r['Won'],
                        'BSP': r['BSP'],
                        'Date': race_date,
                        'TimeLag1': time_lag1,
                        'TimeLag2': time_lag2,
                        'TimeLag3': time_lag3,
                        'TimeAvg3': time_avg3,
                        'TimeAvg5': time_avg5,
                        'TimeStd': time_std,
                        'TimeTrend': time_trend,
                        'TimeBest': time_best,
                        'SplitAvg': split_avg,
                        'SplitLag1': split_lag1,
                        'PosAvg': pos_avg,
                        'PosLag1': pos_lag1,
                        'WinRate': win_rate,
                        'PlaceRate': place_rate,
                        'DaysSince': days_since,
                        'RaceCount': race_count,
                        'Box': r['Box'],
                        'Distance': distance,
                        'Tier': tier
                    })
            
            if dog_id not in dog_history:
                dog_history[dog_id] = []
            dog_history[dog_id].append((
                race_date, 
                r['NormTime'] if pd.notna(r['NormTime']) else None,
                r['Split'] if pd.notna(r['Split']) else None,
                r['Position']
            ))
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    print(f"  Features: {len(feature_rows):,}")
    
    print("\n[4/7] Preparing train/test split...")
    
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    
    feat_df = feat_df[(feat_df['TimeAvg5'] > -5) & (feat_df['TimeAvg5'] < 5)]
    
    train_df = feat_df[(feat_df['Date'] >= datetime(2020, 1, 1)) & (feat_df['Date'] < datetime(2024, 1, 1))]
    test_df = feat_df[feat_df['Date'] >= datetime(2024, 1, 1)]
    
    if len(train_df) > 300000:
        print(f"  Subsampling from {len(train_df):,} to 300,000...")
        train_df = train_df.sample(n=300000, random_state=42)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    feature_cols = ['TimeLag1', 'TimeLag2', 'TimeLag3', 'TimeAvg3', 'TimeAvg5', 
                    'TimeStd', 'TimeTrend', 'TimeBest', 'SplitAvg', 'SplitLag1',
                    'PosAvg', 'PosLag1', 'WinRate', 'PlaceRate', 
                    'DaysSince', 'RaceCount', 'Box', 'Distance', 'Tier']
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Won']
    
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[6/7] Training with LITE GridSearchCV...")
    
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
    
    print("\n[7/7] Evaluating on test set...")
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print(f"\nTest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nML Pace Top Pick:")
    print(f"  Total races: {len(race_leaders):,}")
    print(f"  Wins: {race_leaders['Won'].sum():,}")
    print(f"  Strike Rate: {race_leaders['Won'].mean()*100:.1f}%")
    
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(12).to_string(index=False))
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS (2024-2025)")
    print("="*70)
    
    def backtest(df, label):
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        valid_bsp = df.dropna(subset=['BSP'])
        if len(valid_bsp) == 0:
            print(f"{label}: No BSP data")
            return
        returns = valid_bsp[valid_bsp['Won'] == 1]['BSP'].sum()
        profit = returns - len(valid_bsp)
        roi = profit / len(valid_bsp) * 100
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
    with open('models/pace_gb_model_lite.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved to models/pace_gb_model_lite.pkl")
    print("="*70)

if __name__ == "__main__":
    train_pace_model()
