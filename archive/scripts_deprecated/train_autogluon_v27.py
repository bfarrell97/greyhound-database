"""
Pace Model V12 - Optimized Feature Set
=======================================
Uses top 69 features from Feature Factory (80% cumulative importance)
Pruned from 174 candidates based on LightGBM feature importance ranking.

Key improvements over V11:
- Reduced noise from low-importance features
- Faster training with focused feature set
- Uses proven high-value features only
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

# Top 69 features from Feature Factory (80% cumulative importance)
TOP_FEATURES = [
    'TrackBoxWinRate', 'DistAvgPos', 'Weight_x_Distance', 'BloodlineVsDog', 'TrackAvgPos',
    'TrainerTrackWinRate', 'BeyerStd', 'DaysSinceRace', 'WeightChange', 'SplitBest',
    'TimeQ25', 'SplitLag1', 'TimeBestRecent3', 'ExperiencePerAge', 'Age_x_Experience',
    'CareerStarts', 'CareerWinRate', 'DistExperience', 'TierExperience', 'TrainerDistWinRate',
    'SplitStd', 'TimeLag1', 'TrainerWinRate', 'LastWonDaysAgo', 'SplitAvg',
    'TrainerTrackRuns', 'TrainerFormVsAll', 'DamRuns', 'TimeStd', 'Trainer_x_Track',
    'DistWinRate', 'RaceFrequency30d', 'Time_x_Trainer', 'TrackExperience', 'TrainerStarts60d',
    'TimeAvg3', 'SpecialistScore', 'WeightStd', 'WeightAvg', 'SireWinRate',
    'Box', 'BoxPreference', 'TimeLag2', 'TimeTrend3', 'BeyerLag1',
    'CareerPlaces', 'PosAvg', 'BloodlineScore', 'TrainerWinRate60d', 'Time_x_TrainerForm',
    'Beyer_x_Trainer', 'DamWinRate', 'ThisBoxWinRate', 'TrackPlaceRate', 'DistPlaceRate',
    'TimeBest', 'TimeImproving', 'TimeAvg', 'WinRate5', 'Bloodline_x_Age',
    'WinsPerAge', 'TimeWorst', 'PosImprovement', 'TimeLag3', 'TrainerWinRate30d',
    'RaceFrequency60d', 'Form_x_Trainer', 'Rest_x_Form', 'CareerWins'
]

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def train_pace_v12():
    print("="*70)
    print("PACE MODEL V12 - OPTIMIZED FEATURE SET (69 TOP FEATURES)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/6] Loading data...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
           ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
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
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
                'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
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
    
    print("\n[2/6] Building lookups...")
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_recent = defaultdict(list)
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
    print("\n[3/6] Building features (69 selected)...")
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        track_name = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                hist = dog_history.get(dog_id, [])
                
                if len(hist) >= 3:
                    recent = hist[-10:]
                    
                    times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    beyers = [h['beyer'] for h in recent if h['beyer'] is not None]
                    positions = [h['position'] for h in recent if h['position'] is not None]
                    closings = [h['closing'] for h in recent if h['closing'] is not None]
                    deltas = [h['pos_delta'] for h in recent if h['pos_delta'] is not None]
                    weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
                    splits = [h['split'] for h in recent if h['split'] is not None]
                    
                    if len(times) >= 3:
                        features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Tier': tier}
                        
                        # Time features
                        features['TimeBest'] = min(times)
                        features['TimeWorst'] = max(times)
                        features['TimeAvg'] = np.mean(times)
                        features['TimeAvg3'] = np.mean(times[-3:])
                        features['TimeLag1'] = times[-1]
                        features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                        features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                        features['TimeStd'] = np.std(times) if len(times) >= 3 else 0
                        features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                        features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                        features['TimeBestRecent3'] = min(times[-3:])
                        features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                        
                        # Split features
                        features['SplitBest'] = min(splits) if splits else 0
                        features['SplitAvg'] = np.mean(splits) if splits else 0
                        features['SplitLag1'] = splits[-1] if splits else 0
                        features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                        
                        # Beyer features
                        features['BeyerLag1'] = beyers[-1] if beyers else 77
                        features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                        
                        # Position features
                        features['PosAvg'] = np.mean(positions)
                        features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                        features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                        features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                        features['CareerStarts'] = min(len(hist), 100)
                        features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                        
                        # Form trend
                        if len(positions) >= 5:
                            first_half = np.mean(positions[:len(positions)//2])
                            second_half = np.mean(positions[len(positions)//2:])
                            form_trend = first_half - second_half
                        else:
                            form_trend = 0
                        
                        features['LastWonDaysAgo'] = 999
                        for i, h in enumerate(reversed(hist)):
                            if h['position'] == 1:
                                features['LastWonDaysAgo'] = (race_date - h['date']).days
                                break
                        
                        # Trainer features
                        trainer_id = r['TrainerID']
                        t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                        features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                        
                        cutoff_30d = race_date - timedelta(days=30)
                        cutoff_60d = race_date - timedelta(days=60)
                        t_rec = trainer_recent.get(trainer_id, [])
                        rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                        rec_60 = [x for x in t_rec if x[0] >= cutoff_60d]
                        
                        features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                        features['TrainerWinRate60d'] = safe_div(sum(x[1] for x in rec_60), len(rec_60), features['TrainerWinRate']) if len(rec_60) >= 10 else features['TrainerWinRate']
                        features['TrainerStarts60d'] = len(rec_60)
                        features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                        
                        t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                        features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                        features['TrainerTrackRuns'] = min(t_track['runs'], 100) / 100
                        
                        dist_key = round(distance / 100) * 100
                        t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                        features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                        
                        # Track features
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                        features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                        features['DistExperience'] = min(len(dist_runs), 30) / 30
                        features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                        
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), 0.35) if len(track_runs) >= 3 else 0.35
                        features['TrackExperience'] = min(len(track_runs), 20) / 20
                        features['TrackAvgPos'] = np.mean([t['position'] for t in track_runs]) if track_runs else features['PosAvg']
                        
                        tier_runs = [h for h in hist if h.get('tier', 0) == tier]
                        features['TierExperience'] = min(len(tier_runs), 30) / 30
                        
                        box = int(r['Box'])
                        tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                        features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                        
                        # Age features
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                        features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                        
                        # Closing features
                        deltas_array = deltas if deltas else [0]
                        features['PosImprovement'] = np.mean(deltas_array)
                        
                        # Bloodline features
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                        features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                        features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                        features['DamRuns'] = min(dam_data['runs'], 200) / 200
                        features['BloodlineVsDog'] = features['BloodlineScore'] - features['CareerWinRate']
                        
                        # Weight features
                        weight_avg = np.mean(weights) if weights else 30
                        current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                        features['WeightAvg'] = weight_avg
                        features['WeightChange'] = current_weight - weight_avg
                        features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                        
                        # Box features
                        features['Box'] = box
                        inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                        outside_runs = [h for h in hist if h.get('box', 4) > 4]
                        inside_rate = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['CareerWinRate']) if len(inside_runs) >= 3 else features['CareerWinRate']
                        outside_rate = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['CareerWinRate']) if len(outside_runs) >= 3 else features['CareerWinRate']
                        features['BoxPreference'] = inside_rate - outside_rate
                        
                        this_box_runs = [h for h in hist if h.get('box', 0) == box]
                        features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['CareerWinRate']) if len(this_box_runs) >= 2 else features['CareerWinRate']
                        
                        # Rest features
                        days_since = (race_date - hist[-1]['date']).days
                        features['DaysSinceRace'] = days_since
                        features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                        features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                        
                        # Derived features
                        features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                        features['Time_x_TrainerForm'] = features['TimeBest'] * features['TrainerWinRate30d']
                        features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                        features['Trainer_x_Track'] = features['TrainerWinRate'] * features['TrainerTrackWinRate']
                        features['Age_x_Experience'] = (age_months / 48) * (features['CareerStarts'] / 100)
                        features['Form_x_Trainer'] = form_trend * features['TrainerWinRate']
                        features['Weight_x_Distance'] = features['WeightChange'] * (1 if distance > 500 else -1)
                        features['Bloodline_x_Age'] = features['BloodlineScore'] * (1 - abs(age_months - 30) / 20)
                        rest_score = max(0, 1 - abs(days_since - 10) / 20)
                        features['Rest_x_Form'] = rest_score * form_trend
                        features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrainerTrackWinRate']) / 3
                        
                        feature_rows.append(features)
        
        # Update lookups
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            box = int(r['Box']) if pd.notna(r['Box']) else 4
            won = r['Won']
            
            dog_history[dog_id].append({
                'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
                'position': r['Position'], 'track_id': track_id, 'distance': distance, 'tier': tier,
                'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
                'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
                'weight': r['Weight'], 'box': box, 'split': r['Split'] if pd.notna(r['Split']) else None
            })
            
            if pd.notna(r['TrainerID']):
                tid = r['TrainerID']
                trainer_all[tid]['runs'] += 1
                if won: trainer_all[tid]['wins'] += 1
                trainer_recent[tid].append((race_date, won, track_id, distance))
                cutoff = race_date - timedelta(days=120)
                trainer_recent[tid] = [x for x in trainer_recent[tid] if x[0] >= cutoff]
                trainer_track[tid][track_id]['runs'] += 1
                if won: trainer_track[tid][track_id]['wins'] += 1
                dist_key = round(distance / 100) * 100
                trainer_dist[tid][dist_key]['runs'] += 1
                if won: trainer_dist[tid][dist_key]['wins'] += 1
            
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
    
    print(f"  Total: {processed:,} races")
    print(f"  Features: {len(feature_rows):,}")
    
    print("\n[4/6] Preparing data...")
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    feat_df = feat_df[(feat_df['TimeBest'] > -5) & (feat_df['TimeBest'] < 5)]
    
    meta_cols = ['RaceID', 'Won', 'BSP', 'Tier']
    feature_cols = [c for c in TOP_FEATURES if c in feat_df.columns]
    print(f"Using {len(feature_cols)} features (from top 69)")
    
    train_df = feat_df[feat_df['RaceID'] % 3 != 0]
    test_df = feat_df[feat_df['RaceID'] % 3 == 0]
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['Won']
    
    # --- FEATURE SELECTION (V27) ---
    print("\n[V27] Performing Feature Selection (Top 50%)...")
    from autogluon.tabular import TabularPredictor
    
    # Train a quick proxy model to estimate importance
    select_path = 'models/ag_selector_proxy'
    proxy_predictor = TabularPredictor(label='Won', path=select_path, verbosity=0).fit(
        train_df.sample(n=min(50000, len(train_df)), random_state=42)[feature_cols + ['Won']],
        time_limit=120, # 2 mins
        hyperparameters={'GBM':{}}
    )
    
    # Get Importance
    fi = proxy_predictor.feature_importance(
        test_df.sample(n=min(10000, len(test_df)), random_state=42)[feature_cols + ['Won']]
    )
    
    # Select Top 50%
    median_imp = fi['importance'].median()
    top_features = fi[fi['importance'] >= median_imp].index.tolist()
    # Ensure at least 10 features
    if len(top_features) < 10:
        top_features = fi.index[:10].tolist()
        
    print(f"Dropped {len(feature_cols) - len(top_features)} features.")
    print(f"Selected {len(top_features)} features: {top_features}")
    
    # Update Feature Cols
    feature_cols = top_features

    # --- AUTOGLUON TRAINING (Main) ---
    print("\n[5/6] Training AutoGluon V27 (Optimized)...")
    
    save_path = 'models/autogluon_pace_v27'
    
    # Ensure binary classification
    predictor = TabularPredictor(
        label='Won', 
        path=save_path, 
        problem_type='binary', 
        eval_metric='log_loss'
    ).fit(
        train_df[feature_cols + ['Won']],
        time_limit=900,
        presets='medium_quality'
    )
    
    # --- EVALUATION ---
    print("\n" + "="*70)
    print("V27 ACCURACY REPORT (Feature Selection Optimized)")
    print("="*70)
    
    # 1. Probability Predictions
    probs = predictor.predict_proba(test_df[feature_cols])
    test_df = test_df.copy()
    test_df['PredProb'] = probs[1] # Probability of winning
    
    # 2. Rated Price Conversion
    test_df['RatedPrice'] = 1.0 / test_df['PredProb']
    
    # 3. Price Accuracy (vs BSP)
    valid = test_df[(test_df['BSP'] > 1) & (test_df['BSP'] < 50)].copy()
    
    valid['Error'] = np.abs(valid['RatedPrice'] - valid['BSP'])
    valid['PctError'] = valid['Error'] / valid['BSP']
    
    mape = valid['PctError'].mean() * 100
    corr = valid[['RatedPrice', 'BSP']].corr().iloc[0,1]
    
    print(f"Evaluated on {len(valid):,} rows (BSP < 50)")
    print(f"MAPE (Rated vs BSP): {mape:.2f}%")
    print(f"Correlation: {corr:.4f}")
    
    print("\nAccuracy by BSP Range:")
    ranges = [(0,5), (5,10), (10,20), (20,50)]
    for low, high in ranges:
        subset = valid[(valid['BSP'] >= low) & (valid['BSP'] < high)]
        if len(subset) > 0:
            sub_mape = subset['PctError'].mean() * 100
            print(f"  ${low}-${high}: MAPE {sub_mape:.2f}% ({len(subset)} rows)")

    # 4. Betting Simulation (Validation)
    print("\nBetting Validation (Value > 20%):")
    bets = valid[valid['BSP'] > valid['RatedPrice'] * 1.2].copy()
    bets['Profit'] = np.where(bets['Won']==1, (bets['BSP'] - 1), -1)
    roi = bets['Profit'].sum() / len(bets) * 100 if len(bets) > 0 else 0
    print(f"  Bets: {len(bets):,}")
    print(f"  ROI: {roi:.2f}%")

if __name__ == "__main__":
    train_pace_v12()
