"""
Enhanced Pace Model with GradientBoosting
==========================================
Builds pace features from scratch, trains GradientBoostingClassifier to predict wins.
Features: Normalized times, rolling averages, sectionals, box position, distance, etc.
GridSearchCV for hyperparameter optimization.
Train: 2020-2023, Test: 2024-2025
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

# Track tiers
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
    print("ENHANCED PACE MODEL WITH GRADIENT BOOSTING")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load benchmarks
    print("\n[1/7] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    print(f"Loaded {len(bench_lookup):,} track/distance benchmarks")
    
    # Load ALL historical race data
    print("\n[2/7] Loading race data...")
    query = """
    SELECT ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, ge.RunningPosition, ge.Margin,
           r.Distance, rm.MeetingDate, t.TrackName, ge.IncomingGrade
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2019-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, ge.RaceID, ge.GreyhoundID
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
    df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    
    # Normalize finish time using benchmarks
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    
    print(f"Loaded {len(df):,} entries")
    print(f"With valid times: {df['NormTime'].notna().sum():,}")
    
    # [3/7] Build historical features for each dog
    print("\n[3/7] Building historical pace features...")
    
    # Sort by dog and date
    df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
    
    # Calculate rolling features per dog
    dog_history = {}  # {dog_id: list of (date, norm_time, split, position, distance, track_tier)}
    
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        # Build features for each dog in this race
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            
            # Get dog's history (before this race)
            hist = dog_history.get(dog_id, [])
            
            if len(hist) >= 3:  # Need at least 3 races for meaningful features
                # Recent times
                recent_times = [h[1] for h in hist[-5:] if h[1] is not None and not np.isnan(h[1])]
                recent_splits = [h[2] for h in hist[-5:] if h[2] is not None and not np.isnan(h[2])]
                recent_positions = [h[3] for h in hist[-5:] if h[3] is not None]
                recent_dist = [h[4] for h in hist[-5:]]
                
                if len(recent_times) >= 3:
                    # Time features
                    time_lag1 = recent_times[-1] if len(recent_times) >= 1 else 0
                    time_lag2 = recent_times[-2] if len(recent_times) >= 2 else time_lag1
                    time_lag3 = recent_times[-3] if len(recent_times) >= 3 else time_lag2
                    time_avg3 = np.mean(recent_times[-3:])
                    time_avg5 = np.mean(recent_times[-5:]) if len(recent_times) >= 5 else time_avg3
                    time_std = np.std(recent_times[-5:]) if len(recent_times) >= 3 else 0
                    time_trend = time_lag1 - time_lag3  # Negative = improving
                    time_best = min(recent_times) if recent_times else 0
                    
                    # Split features
                    split_avg = np.mean(recent_splits) if recent_splits else 0
                    split_lag1 = recent_splits[-1] if recent_splits else 0
                    
                    # Position features
                    pos_avg = np.mean(recent_positions) if recent_positions else 4
                    pos_lag1 = recent_positions[-1] if recent_positions else 4
                    win_rate = sum(1 for p in recent_positions if p == 1) / len(recent_positions) if recent_positions else 0
                    place_rate = sum(1 for p in recent_positions if p <= 3) / len(recent_positions) if recent_positions else 0
                    
                    # Days since last race
                    last_race_date = hist[-1][0]
                    days_since = (race_date - last_race_date).days
                    days_since = min(days_since, 90)
                    
                    # Experience
                    race_count = min(len(hist), 50)
                    
                    # Distance suitability
                    dist_matches = sum(1 for d in recent_dist if abs(d - distance) < 50)
                    dist_experience = dist_matches / 5 if recent_dist else 0
                    
                    # Tier experience
                    tier_races = [h[5] for h in hist[-10:]]
                    metro_exp = sum(1 for t in tier_races if t == 2) / len(tier_races) if tier_races else 0
                    
                    feature_rows.append({
                        'RaceID': race_id,
                        'GreyhoundID': dog_id,
                        'Won': r['Won'],
                        'BSP': r['BSP'],
                        'Date': race_date,
                        # Time features
                        'TimeLag1': time_lag1,
                        'TimeLag2': time_lag2,
                        'TimeLag3': time_lag3,
                        'TimeAvg3': time_avg3,
                        'TimeAvg5': time_avg5,
                        'TimeStd': time_std,
                        'TimeTrend': time_trend,
                        'TimeBest': time_best,
                        # Split features
                        'SplitAvg': split_avg,
                        'SplitLag1': split_lag1,
                        # Position features
                        'PosAvg': pos_avg,
                        'PosLag1': pos_lag1,
                        'WinRate': win_rate,
                        'PlaceRate': place_rate,
                        # Other features
                        'DaysSince': days_since,
                        'RaceCount': race_count,
                        'DistExperience': dist_experience,
                        'MetroExp': metro_exp,
                        'Box': r['Box'],
                        'Distance': distance,
                        'Tier': tier
                    })
            
            # Add this race to dog's history
            if dog_id not in dog_history:
                dog_history[dog_id] = []
            dog_history[dog_id].append((
                race_date, 
                r['NormTime'] if pd.notna(r['NormTime']) else None,
                r['Split'] if pd.notna(r['Split']) else None,
                r['Position'],
                r['Distance'],
                tier
            ))
        
        processed += 1
        if processed % 20000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    print(f"  Features generated: {len(feature_rows):,}")
    
    # [4/7] Create DataFrame and split
    print("\n[4/7] Preparing train/test split...")
    
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    
    # Remove outliers
    feat_df = feat_df[feat_df['TimeAvg5'] > -5]  # Not too fast
    feat_df = feat_df[feat_df['TimeAvg5'] < 5]   # Not too slow
    
    # Train: 2020-2023, Test: 2024-2025
    train_df = feat_df[(feat_df['Date'] >= datetime(2020, 1, 1)) & (feat_df['Date'] < datetime(2024, 1, 1))]
    test_df = feat_df[feat_df['Date'] >= datetime(2024, 1, 1)]
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    # Feature columns
    feature_cols = ['TimeLag1', 'TimeLag2', 'TimeLag3', 'TimeAvg3', 'TimeAvg5', 
                    'TimeStd', 'TimeTrend', 'TimeBest', 'SplitAvg', 'SplitLag1',
                    'PosAvg', 'PosLag1', 'WinRate', 'PlaceRate', 
                    'DaysSince', 'RaceCount', 'DistExperience', 'MetroExp',
                    'Box', 'Distance', 'Tier']
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Won']
    
    # [5/7] Scale features
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # [6/7] Train with GridSearchCV
    print("\n[6/7] Training GradientBoostingClassifier with GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [50, 100],
        'subsample': [0.8, 1.0]
    }
    
    print("Grid search parameters:", param_grid)
    print("This may take a while...")
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    # [7/7] Evaluate
    print("\n[7/7] Evaluating on test set...")
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    
    # Find pace leader per race (highest win probability)
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print(f"\nTest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nML Pace Top Pick (highest prob per race):")
    print(f"  Total races: {len(race_leaders):,}")
    print(f"  Wins: {race_leaders['Won'].sum():,}")
    print(f"  Strike Rate: {race_leaders['Won'].mean()*100:.1f}%")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(15).to_string(index=False))
    
    # Backtest
    print("\n" + "="*70)
    print("BACKTEST RESULTS (2024-2025)")
    print("="*70)
    
    def backtest(df, label):
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        valid_bsp = df.dropna(subset=['BSP'])
        if len(valid_bsp) == 0:
            print(f"{label}: No valid BSP data")
            return
        returns = valid_bsp[valid_bsp['Won'] == 1]['BSP'].sum()
        profit = returns - len(valid_bsp)
        roi = profit / len(valid_bsp) * 100
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- ALL TRACKS ---")
    backtest(race_leaders, "All picks")
    f = race_leaders[(race_leaders['BSP'] >= 2) & (race_leaders['BSP'] <= 10)]
    backtest(f, "$2-$10")
    f = race_leaders[(race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)]
    backtest(f, "$3-$8")
    
    print("\n--- BY TIER ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        t = race_leaders[race_leaders['Tier'] == tier]
        if len(t) > 100:
            t_filt = t[(t['BSP'] >= 2) & (t['BSP'] <= 10)]
            backtest(t_filt, f"{name} $2-$10")
    
    print("\n--- BY CONFIDENCE ---")
    high_conf = race_leaders[race_leaders['PredProb'] >= race_leaders['PredProb'].quantile(0.75)]
    backtest(high_conf[(high_conf['BSP'] >= 2) & (high_conf['BSP'] <= 10)], "Top 25% confidence")
    
    # Save model
    print("\n" + "="*70)
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_params': grid_search.best_params_
    }
    with open('models/pace_gb_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved model to models/pace_gb_model.pkl")
    print("="*70)

if __name__ == "__main__":
    train_pace_model()
