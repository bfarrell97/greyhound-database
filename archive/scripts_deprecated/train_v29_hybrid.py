"""
V29 - Hybrid Model (V28 Rolling Window + V12 Pace Features)
============================================================
Combines:
- V28: Rolling window features (28D, 91D, 365D) from Betfair Tutorial
- V12: Top 69 pace/trainer/bloodline features

Then removes redundant/highly-correlated features before training.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import itertools
import warnings
warnings.filterwarnings('ignore')

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def remove_correlated_features(df, feature_cols, threshold=0.95):
    """Remove features that are highly correlated with each other."""
    print(f"  Starting with {len(feature_cols)} features...")
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find pairs with correlation > threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    remaining = [f for f in feature_cols if f not in to_drop]
    print(f"  Removed {len(to_drop)} highly correlated features (>{threshold})")
    print(f"  Remaining: {len(remaining)} features")
    
    return remaining

def train_v29():
    print("="*70)
    print("V29 - HYBRID MODEL (V28 Rolling + V12 Pace Features)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # ===== 1. LOAD DATA =====
    print("\n[1/8] Loading data...")
    
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    query = """
    SELECT 
        ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
        ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight, ge.PrizeMoney,
        ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
        ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
        g.SireID, g.DamID, g.DateOfBirth
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} entries")
    
    # ===== 2. BASIC PREPROCESSING =====
    print("\n[2/8] Preprocessing...")
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight', 'PrizeMoney',
                'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    
    # V28 normalized features
    df['Prizemoney_norm'] = np.log10(df['PrizeMoney'].fillna(0) + 1) / 12
    df['Place_inv'] = (1 / df['Position']).fillna(0)
    df['Place_log'] = np.log10(df['Position'] + 1).fillna(0)
    
    # ===== 3. V28 ROLLING WINDOW FEATURES =====
    print("\n[3/8] Generating V28 rolling window features...")
    
    # Reference times
    win_results = df[df['Won'] == 1]
    median_win_time = win_results[win_results['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    median_win_time.columns = ['TrackName', 'Distance', 'RunTime_median']
    median_win_split = win_results[win_results['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    median_win_split.columns = ['TrackName', 'Distance', 'SplitMargin_median']
    
    df = df.merge(median_win_time, on=['TrackName', 'Distance'], how='left')
    df = df.merge(median_win_split, on=['TrackName', 'Distance'], how='left')
    
    df['RunTime_norm'] = (df['RunTime_median'] / df['FinishTime']).clip(0.9, 1.1).fillna(1.0)
    df['SplitMargin_norm'] = (df['SplitMargin_median'] / df['Split']).clip(0.9, 1.1).fillna(1.0)
    
    # Box win percent
    box_win_percent = df.groupby(['TrackName', 'Distance', 'Box'])['Won'].mean().reset_index()
    box_win_percent.columns = ['TrackName', 'Distance', 'Box', 'box_win_percent']
    df = df.merge(box_win_percent, on=['TrackName', 'Distance', 'Box'], how='left')
    
    # Rolling windows
    dataset = df.set_index(['GreyhoundID', 'MeetingDate']).sort_index()
    
    rolling_windows = ['28D', '91D', '365D']
    features_to_roll = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm']
    aggregates = ['mean', 'std']  # Reduced aggregates to save memory
    v28_feature_cols = ['box_win_percent']
    
    for rolling_window in rolling_windows:
        print(f'  Processing {rolling_window}...')
        rolling_result = (
            dataset.reset_index(level=0)
            .groupby('GreyhoundID')[features_to_roll]
            .rolling(rolling_window)
            .agg(aggregates)
            .groupby(level=0)
            .shift(1)
        )
        agg_cols = [f'{f}_{a}_{rolling_window}' for f, a in itertools.product(features_to_roll, aggregates)]
        dataset[agg_cols] = rolling_result
        v28_feature_cols.extend(agg_cols)
    
    dataset = dataset.reset_index()
    dataset.fillna(0, inplace=True)
    
    # ===== 4. V12 PACE FEATURES =====
    print("\n[4/8] Generating V12 pace features...")
    
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
    v12_features = []
    processed = 0
    
    for race_id, race_df in dataset.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        # Only generate features for test period
        if race_date >= datetime(2024, 1, 1):
            for idx, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                hist = dog_history.get(dog_id, [])
                
                if len(hist) >= 3:
                    recent = hist[-10:]
                    times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    positions = [h['position'] for h in recent if h['position'] is not None]
                    
                    if len(times) >= 3:
                        features = {'idx': idx}
                        
                        # Key V12 features (subset to avoid redundancy with V28)
                        features['TimeBest'] = min(times)
                        features['TimeAvg'] = np.mean(times)
                        features['TimeStd'] = np.std(times) if len(times) >= 3 else 0
                        features['PosAvg'] = np.mean(positions)
                        features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                        features['CareerStarts'] = min(len(hist), 100)
                        features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                        
                        # Trainer features
                        trainer_id = r['TrainerID']
                        t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                        features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                        
                        t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                        features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                        
                        # Track/Distance features
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                        
                        # Bloodline
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                        features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                        
                        # Box
                        box = int(r['Box'])
                        tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                        features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                        
                        v12_features.append(features)
        
        # Update lookups
        for idx, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            box = int(r['Box']) if pd.notna(r['Box']) else 4
            won = r['Won']
            
            dog_history[dog_id].append({
                'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'position': r['Position'], 'track_id': track_id, 'distance': distance
            })
            
            if pd.notna(r['TrainerID']):
                tid = r['TrainerID']
                trainer_all[tid]['runs'] += 1
                if won: trainer_all[tid]['wins'] += 1
                trainer_track[tid][track_id]['runs'] += 1
                if won: trainer_track[tid][track_id]['wins'] += 1
            
            if pd.notna(r['SireID']):
                sire_stats[r['SireID']]['runs'] += 1
                if won: sire_stats[r['SireID']]['wins'] += 1
            if pd.notna(r['DamID']):
                dam_stats[r['DamID']]['runs'] += 1
                if won: dam_stats[r['DamID']]['wins'] += 1
            
            track_box_wins[track_id][box]['runs'] += 1
            if won: track_box_wins[track_id][box]['wins'] += 1
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Generated V12 features for {len(v12_features):,} entries")
    
    # ===== 5. MERGE FEATURES =====
    print("\n[5/8] Merging V28 + V12 features...")
    
    v12_df = pd.DataFrame(v12_features).set_index('idx')
    v12_feature_cols = [c for c in v12_df.columns]
    
    # Merge V12 into dataset
    for col in v12_feature_cols:
        dataset[col] = np.nan
    dataset.loc[v12_df.index, v12_feature_cols] = v12_df[v12_feature_cols]
    
    # Combined feature list
    all_feature_cols = v28_feature_cols + v12_feature_cols
    all_feature_cols = list(set(all_feature_cols))  # Remove duplicates
    
    print(f"  Total features before dedup: {len(all_feature_cols)}")
    
    # ===== 6. REMOVE REDUNDANT FEATURES =====
    print("\n[6/8] Removing redundant features...")
    
    # Filter to rows with V12 features (test period)
    model_df = dataset[dataset['TimeBest'].notna()].copy()
    model_df = model_df[['RaceID', 'Won', 'BSP'] + all_feature_cols]
    model_df = model_df.dropna()
    
    final_features = remove_correlated_features(model_df, all_feature_cols, threshold=0.90)
    
    # ===== 7. TRAIN MODEL =====
    print("\n[7/8] Training AutoGluon...")
    
    # Split 80/20
    split_date = dataset[dataset['TimeBest'].notna()]['MeetingDate'].quantile(0.8)
    train_df = model_df[dataset.loc[model_df.index, 'MeetingDate'] < split_date]
    test_df = model_df[dataset.loc[model_df.index, 'MeetingDate'] >= split_date]
    
    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    from autogluon.tabular import TabularPredictor
    
    predictor = TabularPredictor(
        label='Won',
        path='models/autogluon_v29_hybrid',
        problem_type='binary',
        eval_metric='log_loss'
    ).fit(
        train_df[final_features + ['Won']],
        time_limit=600,
        presets='medium_quality'
    )
    
    # ===== 8. EVALUATE =====
    print("\n" + "="*70)
    print("V29 HYBRID EVALUATION REPORT")
    print("="*70)
    
    probs = predictor.predict_proba(test_df[final_features])
    test_df = test_df.copy()
    test_df['PredProb'] = probs[1]
    test_df['RatedPrice'] = 1 / test_df['PredProb']
    
    # Price accuracy
    valid = test_df[(test_df['BSP'] > 1) & (test_df['BSP'] < 50)].copy()
    valid['PctError'] = np.abs(valid['RatedPrice'] - valid['BSP']) / valid['BSP']
    mape = valid['PctError'].mean() * 100
    corr = valid[['RatedPrice', 'BSP']].corr().iloc[0,1]
    
    print(f"Features used: {len(final_features)}")
    print(f"MAPE (vs BSP): {mape:.2f}%")
    print(f"Correlation: {corr:.4f}")
    
    # Betting with 10% commission
    COMMISSION = 0.10
    print(f"\nBetting Validation (Value > 20%, {COMMISSION*100:.0f}% commission):")
    bets = valid[valid['BSP'] > valid['RatedPrice'] * 1.2].copy()
    bets['Profit'] = np.where(bets['Won']==1, (bets['BSP'] - 1) * (1 - COMMISSION), -1)
    roi = bets['Profit'].sum() / len(bets) * 100 if len(bets) > 0 else 0
    wins = bets['Won'].sum()
    strike_rate = wins / len(bets) * 100 if len(bets) > 0 else 0
    print(f"  Bets: {len(bets):,}")
    print(f"  Wins: {wins:,} ({strike_rate:.1f}%)")
    print(f"  ROI (after commission): {roi:.2f}%")

if __name__ == "__main__":
    train_v29()
