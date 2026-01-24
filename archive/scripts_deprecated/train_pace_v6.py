"""
Pace Model V6 - Roll3/Roll5, Prize Money, Culled Features
==========================================================
Builds on V5 best features + adds:
1. Roll3/Roll5 for time, position, Beyer
2. Prize money features (total, recent, trend)
3. Cull low-importance features from V5
Goal: Maximize strike rate.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
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

def train_v6_model():
    print("="*70)
    print("PACE MODEL V6 - ROLL3/ROLL5 + PRIZE MONEY")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/7] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    print("\n[2/7] Loading data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.PrizeMoney,
           ge.FirstSplitPosition, ge.SecondSplitTime,
           ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
           g.SireID, g.DamID, g.DateOfBirth
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
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
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'PrizeMoney',
                'FirstSplitPosition', 'SecondSplitTime']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
    df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[3/7] Building lookups...")
    
    dog_history = defaultdict(list)
    trainer_recent = defaultdict(lambda: {'wins': 0, 'runs': 0})
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    
    print("\n[4/7] Building features...")
    
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                hist = dog_history.get(dog_id, [])
                
                if len(hist) >= 3:
                    # Get last 10 races for various windows
                    recent = hist[-10:]
                    recent_times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    recent_beyer = [h['beyer'] for h in recent if h['beyer'] is not None]
                    recent_pos = [h['position'] for h in recent if h['position'] is not None]
                    recent_prizes = [h['prize'] for h in recent if h['prize'] is not None and h['prize'] > 0]
                    
                    if len(recent_times) >= 3:
                        # === ROLL3/ROLL5 FEATURES ===
                        roll3_time = np.mean(recent_times[-3:])
                        roll5_time = np.mean(recent_times[-5:]) if len(recent_times) >= 5 else roll3_time
                        roll3_pos = np.mean(recent_pos[-3:]) if len(recent_pos) >= 3 else 4
                        roll5_pos = np.mean(recent_pos[-5:]) if len(recent_pos) >= 5 else roll3_pos
                        roll3_beyer = np.mean(recent_beyer[-3:]) if len(recent_beyer) >= 3 else 77
                        roll5_beyer = np.mean(recent_beyer[-5:]) if len(recent_beyer) >= 5 else roll3_beyer
                        
                        # Best times
                        time_best = min(recent_times)
                        beyer_best = max(recent_beyer) if recent_beyer else 77
                        
                        # Win/place rates
                        win_rate = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                        
                        # Trainer
                        trainer_id = r['TrainerID']
                        trainer_data = trainer_recent.get(trainer_id, {'wins': 0, 'runs': 0})
                        trainer_win_rate = trainer_data['wins'] / trainer_data['runs'] if trainer_data['runs'] > 20 else 0.12
                        
                        # Distance specialist
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        dist_win_rate = sum(1 for d in dist_runs if d['position'] == 1) / len(dist_runs) if len(dist_runs) >= 3 else win_rate
                        dist_experience = min(len(dist_runs), 20) / 20
                        
                        # Track specialist
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        track_win_rate = sum(1 for t in track_runs if t['position'] == 1) / len(track_runs) if len(track_runs) >= 3 else win_rate
                        
                        # Bloodline
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        sire_win = sire_data['wins'] / sire_data['runs'] if sire_data['runs'] > 50 else 0.12
                        dam_win = dam_data['wins'] / dam_data['runs'] if dam_data['runs'] > 30 else 0.12
                        bloodline_score = (sire_win + dam_win) / 2
                        
                        # Career/Age
                        career_starts = min(len(hist), 100)
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        age_x_experience = (age_months / 48) * (career_starts / 100)
                        
                        # Box
                        box = r['Box']
                        
                        # === PRIZE MONEY FEATURES ===
                        total_prize = sum(h['prize'] for h in hist if h['prize'] is not None and h['prize'] > 0)
                        last5_prize = sum(recent_prizes[-5:]) if len(recent_prizes) >= 5 else sum(recent_prizes)
                        avg_career_prize = total_prize / len(hist) if len(hist) > 0 else 0
                        avg_recent_prize = np.mean(recent_prizes[-5:]) if len(recent_prizes) >= 5 else (np.mean(recent_prizes) if recent_prizes else 0)
                        prize_trend = avg_recent_prize - avg_career_prize
                        
                        # === KEY INTERACTIONS (KEEP TOP PERFORMERS) ===
                        time_best_x_trainer = time_best * trainer_win_rate
                        beyer_x_trainer = beyer_best * trainer_win_rate
                        time_best_x_dist_exp = time_best * dist_experience
                        win_rate_x_dist = win_rate * dist_win_rate
                        win_rate_x_track = win_rate * track_win_rate
                        
                        # Roll interactions
                        roll5_x_trainer = roll5_beyer * trainer_win_rate
                        
                        feature_rows.append({
                            'RaceID': race_id,
                            'Won': r['Won'],
                            'BSP': r['BSP'],
                            'Tier': tier,
                            # Roll features (6)
                            'Roll3_Time': roll3_time, 'Roll5_Time': roll5_time,
                            'Roll3_Pos': roll3_pos, 'Roll5_Pos': roll5_pos,
                            'Roll3_Beyer': roll3_beyer, 'Roll5_Beyer': roll5_beyer,
                            # Best features (2)
                            'TimeBest': time_best, 'BeyerBest': beyer_best,
                            # Rate features (3)
                            'WinRate': win_rate, 'DistWinRate': dist_win_rate, 'TrackWinRate': track_win_rate,
                            # Connection features (2)
                            'TrainerWinRate': trainer_win_rate, 'BloodlineScore': bloodline_score,
                            # Career/Age (3)
                            'CareerStarts': career_starts, 'AgeMonths': age_months, 'Age_x_Experience': age_x_experience,
                            # Box (1)
                            'Box': box,
                            # Prize money (4)
                            'TotalPrize': total_prize / 10000, 'Last5Prize': last5_prize / 1000,
                            'AvgRecentPrize': avg_recent_prize / 100, 'PrizeTrend': prize_trend / 100,
                            # Interactions (6)
                            'TimeBest_x_Trainer': time_best_x_trainer, 'Beyer_x_Trainer': beyer_x_trainer,
                            'TimeBest_x_DistExp': time_best_x_dist_exp,
                            'WinRate_x_Dist': win_rate_x_dist, 'WinRate_x_Track': win_rate_x_track,
                            'Roll5_x_Trainer': roll5_x_trainer
                        })
        
        # Update histories
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            dog_history[dog_id].append({
                'date': race_date,
                'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
                'position': r['Position'],
                'track_id': track_id,
                'distance': distance,
                'prize': r['PrizeMoney']
            })
            
            if pd.notna(r['TrainerID']):
                trainer_recent[r['TrainerID']]['runs'] += 1
                if r['Won'] == 1:
                    trainer_recent[r['TrainerID']]['wins'] += 1
            
            if pd.notna(r['SireID']):
                sire_stats[r['SireID']]['runs'] += 1
                if r['Won'] == 1:
                    sire_stats[r['SireID']]['wins'] += 1
            if pd.notna(r['DamID']):
                dam_stats[r['DamID']]['runs'] += 1
                if r['Won'] == 1:
                    dam_stats[r['DamID']]['wins'] += 1
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    print(f"  Features: {len(feature_rows):,}")
    
    print("\n[5/7] Preparing data...")
    
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    feat_df = feat_df[(feat_df['TimeBest'] > -5) & (feat_df['TimeBest'] < 5)]
    
    train_df = feat_df[feat_df['RaceID'] % 3 != 0]
    test_df = feat_df[feat_df['RaceID'] % 3 == 0]
    
    if len(train_df) > 300000:
        print(f"  Subsampling from {len(train_df):,} to 300,000...")
        train_df = train_df.sample(n=300000, random_state=42)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    feature_cols = [
        'Roll3_Time', 'Roll5_Time', 'Roll3_Pos', 'Roll5_Pos', 'Roll3_Beyer', 'Roll5_Beyer',
        'TimeBest', 'BeyerBest', 'WinRate', 'DistWinRate', 'TrackWinRate',
        'TrainerWinRate', 'BloodlineScore', 'CareerStarts', 'AgeMonths', 'Age_x_Experience',
        'Box', 'TotalPrize', 'Last5Prize', 'AvgRecentPrize', 'PrizeTrend',
        'TimeBest_x_Trainer', 'Beyer_x_Trainer', 'TimeBest_x_DistExp',
        'WinRate_x_Dist', 'WinRate_x_Track', 'Roll5_x_Trainer'
    ]
    
    print(f"Feature count: {len(feature_cols)}")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Won']
    
    print("\n[6/7] Training...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100],
        'max_depth': [4, 5],
        'learning_rate': [0.1],
        'min_samples_split': [100],
        'subsample': [0.8]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest params: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    print("\n[7/7] Evaluating...")
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print(f"\nTest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nML Top Pick:")
    print(f"  Total races: {len(race_leaders):,}")
    print(f"  Wins: {race_leaders['Won'].sum():,}")
    print(f"  Strike Rate: {race_leaders['Won'].mean()*100:.1f}%")
    
    print("\nTop 15 Feature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(15).to_string(index=False))
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    def backtest(df, label):
        if len(df) < 50:
            return
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        valid = df.dropna(subset=['BSP'])
        returns = valid[valid['Won'] == 1]['BSP'].sum()
        profit = returns - len(valid)
        roi = profit / len(valid) * 100 if len(valid) > 0 else 0
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- ALL TRACKS ---")
    backtest(race_leaders, "All picks")
    backtest(race_leaders[(race_leaders['BSP'] >= 2) & (race_leaders['BSP'] <= 10)], "$2-$10")
    backtest(race_leaders[(race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)], "$3-$8")
    
    print("\n--- BY TIER ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        t = race_leaders[(race_leaders['Tier'] == tier) & (race_leaders['BSP'] >= 2) & (race_leaders['BSP'] <= 10)]
        backtest(t, f"{name} $2-$10")
    
    print("\n" + "="*70)
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_params': grid_search.best_params_
    }
    with open('models/pace_v6_roll_prize.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved to models/pace_v6_roll_prize.pkl")
    print("="*70)

if __name__ == "__main__":
    train_v6_model()
