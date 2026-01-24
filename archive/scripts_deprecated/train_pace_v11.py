"""
Pace Model V11 - Enhanced Trainer Stats & Feature Interactions
===============================================================
Builds on V5 with:
1. Enhanced Trainer Stats:
   - TrainerWinRate_30d (recent form)
   - TrainerWinRate_Track (track specialist)
   - TrainerWinRate_Distance (distance specialist)
   - TrainerStarts_30d (activity level)
   
2. New Feature Interactions:
   - TimeBest_x_TrainerForm
   - WinRate_x_BoxType
   - DistWinRate_x_TrackWinRate
   - Beyer_x_IsPeakAge
   - ClosingSpeed_x_Box
   
3. New Derived Features:
   - DaysSinceRace (rest/freshness)
   - ConsistencyScore (position variance)
   - RecentTrend (improving/declining)
   
Uses LightGBM with GPU acceleration.
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

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def train_v11_model():
    print("="*70)
    print("PACE MODEL V11 - ENHANCED TRAINER STATS & INTERACTIONS")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/7] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    print("\n[2/7] Loading data...")
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
    
    # Calculate Age
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    
    # Calculate Closing Speed (last split)
    df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
    
    # Position Delta (positions gained from first split to finish)
    df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[3/7] Building lookups...")
    
    dog_history = defaultdict(list)
    
    # Enhanced trainer tracking
    trainer_all_time = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_recent = defaultdict(list)  # List of (date, won, track_id, distance) tuples
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    trainer_distance = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
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
                    recent = hist[-10:]
                    recent_times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    recent_beyer = [h['beyer'] for h in recent if h['beyer'] is not None]
                    recent_pos = [h['position'] for h in recent if h['position'] is not None]
                    recent_closing = [h['closing'] for h in recent if h['closing'] is not None]
                    recent_delta = [h['pos_delta'] for h in recent if h['pos_delta'] is not None]
                    recent_weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
                    
                    if len(recent_times) >= 3:
                        # === V5 BASE FEATURES ===
                        time_best = min(recent_times)
                        time_avg3 = np.mean(recent_times[-3:])
                        time_lag1 = recent_times[-1]
                        
                        pos_avg = np.mean(recent_pos) if recent_pos else 4
                        win_rate = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                        
                        # === ENHANCED TRAINER STATS (V11) ===
                        trainer_id = r['TrainerID']
                        
                        # All-time trainer win rate
                        t_all = trainer_all_time.get(trainer_id, {'wins': 0, 'runs': 0})
                        trainer_win_rate = t_all['wins'] / t_all['runs'] if t_all['runs'] > 20 else 0.12
                        
                        # Trainer win rate last 30 days
                        cutoff_30d = race_date - timedelta(days=30)
                        t_recent = trainer_recent.get(trainer_id, [])
                        recent_30d = [(d, w, tid, dist) for d, w, tid, dist in t_recent if d >= cutoff_30d]
                        trainer_win_rate_30d = sum(w for _, w, _, _ in recent_30d) / len(recent_30d) if len(recent_30d) >= 5 else trainer_win_rate
                        trainer_starts_30d = len(recent_30d)
                        
                        # Trainer win rate at this track
                        t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                        trainer_win_rate_track = t_track['wins'] / t_track['runs'] if t_track['runs'] >= 10 else trainer_win_rate
                        
                        # Trainer win rate at this distance (+/- 50m)
                        dist_key = round(distance / 100) * 100  # Round to nearest 100m
                        t_dist = trainer_distance.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                        trainer_win_rate_dist = t_dist['wins'] / t_dist['runs'] if t_dist['runs'] >= 10 else trainer_win_rate
                        
                        # === EXISTING V5 FEATURES ===
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        dist_win_rate = sum(1 for d in dist_runs if d['position'] == 1) / len(dist_runs) if len(dist_runs) >= 3 else win_rate
                        dist_experience = min(len(dist_runs), 20) / 20
                        
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        track_win_rate = sum(1 for t in track_runs if t['position'] == 1) / len(track_runs) if len(track_runs) >= 3 else win_rate
                        
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        sire_win = sire_data['wins'] / sire_data['runs'] if sire_data['runs'] > 50 else 0.12
                        dam_win = dam_data['wins'] / dam_data['runs'] if dam_data['runs'] > 30 else 0.12
                        bloodline_score = (sire_win + dam_win) / 2
                        
                        career_starts = min(len(hist), 100)
                        box = r['Box']
                        box_type = 1 if box <= 4 else 0  # 1 = inside, 0 = outside
                        
                        beyer_best = max(recent_beyer) if recent_beyer else 77
                        beyer_avg = np.mean(recent_beyer) if recent_beyer else 77
                        beyer_lag1 = recent_beyer[-1] if recent_beyer else 77
                        
                        # Age features
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        is_peak_age = 1 if 24 <= age_months <= 36 else 0
                        is_young = 1 if age_months < 24 else 0
                        is_veteran = 1 if age_months > 42 else 0
                        age_x_experience = (age_months / 48) * (career_starts / 100)
                        
                        # Closing speed
                        closing_avg = np.mean(recent_closing) if recent_closing else 0
                        closing_best = min(recent_closing) if recent_closing else 0
                        closing_lag1 = recent_closing[-1] if recent_closing else 0
                        
                        # Position delta
                        delta_avg = np.mean(recent_delta) if recent_delta else 0
                        delta_lag1 = recent_delta[-1] if recent_delta else 0
                        is_closer = 1 if delta_avg > 1 else 0
                        is_leader = 1 if delta_avg < -1 else 0
                        
                        # Weight
                        weight_avg = np.mean(recent_weights) if recent_weights else 30
                        current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                        weight_change = current_weight - weight_avg
                        
                        # === NEW V11 DERIVED FEATURES ===
                        
                        # Days since last race
                        last_race_date = hist[-1]['date']
                        days_since_race = (race_date - last_race_date).days
                        is_fresh = 1 if 7 <= days_since_race <= 14 else 0  # Optimal rest
                        is_backed_up = 1 if days_since_race <= 5 else 0    # Quick backup
                        is_spelled = 1 if days_since_race > 28 else 0      # Long break
                        
                        # Consistency score (lower = more consistent)
                        consistency_score = np.std(recent_pos) if len(recent_pos) >= 3 else 3.0
                        
                        # Recent trend (positive = improving)
                        if len(recent_pos) >= 5:
                            first_half = np.mean(recent_pos[:len(recent_pos)//2])
                            second_half = np.mean(recent_pos[len(recent_pos)//2:])
                            recent_trend = first_half - second_half  # Positive = improving
                        else:
                            recent_trend = 0
                        
                        # === V5 INTERACTIONS ===
                        time_best_x_trainer = time_best * trainer_win_rate
                        beyer_x_trainer = beyer_best * trainer_win_rate
                        time_best_x_dist_exp = time_best * dist_experience
                        win_rate_x_dist = win_rate * dist_win_rate
                        win_rate_x_track = win_rate * track_win_rate
                        closing_x_trainer = closing_avg * trainer_win_rate
                        age_x_beyer = (age_months / 48) * (beyer_avg / 100)
                        
                        # === NEW V11 INTERACTIONS ===
                        time_best_x_trainer_form = time_best * trainer_win_rate_30d
                        win_rate_x_box_type = win_rate * box_type  # Good dogs from inside
                        dist_x_track_specialist = dist_win_rate * track_win_rate
                        beyer_x_peak_age = beyer_best * is_peak_age
                        closing_x_box = closing_avg * (1 - box_type)  # Closers from outside
                        trainer_form_x_dog_form = trainer_win_rate_30d * win_rate
                        consistency_x_win_rate = (1 / (consistency_score + 0.1)) * win_rate
                        trend_x_trainer = recent_trend * trainer_win_rate
                        
                        feature_rows.append({
                            'RaceID': race_id,
                            'Won': r['Won'],
                            'BSP': r['BSP'],
                            'Tier': tier,
                            # V5 Base Features (19)
                            'TimeBest': time_best, 'TimeAvg3': time_avg3, 'TimeLag1': time_lag1,
                            'PosAvg': pos_avg, 'WinRate': win_rate,
                            'TrainerWinRate': trainer_win_rate, 'DistWinRate': dist_win_rate,
                            'TrackWinRate': track_win_rate, 'BloodlineScore': bloodline_score,
                            'CareerStarts': career_starts, 'Box': box,
                            'BeyerBest': beyer_best, 'BeyerAvg': beyer_avg, 'BeyerLag1': beyer_lag1,
                            'TimeBest_x_Trainer': time_best_x_trainer, 'Beyer_x_Trainer': beyer_x_trainer,
                            'TimeBest_x_DistExp': time_best_x_dist_exp,
                            'WinRate_x_Dist': win_rate_x_dist, 'WinRate_x_Track': win_rate_x_track,
                            # V5 Age/Closing/Delta Features (15)
                            'AgeMonths': age_months, 'IsPeakAge': is_peak_age,
                            'IsYoung': is_young, 'IsVeteran': is_veteran, 'Age_x_Experience': age_x_experience,
                            'ClosingAvg': closing_avg, 'ClosingBest': closing_best, 'ClosingLag1': closing_lag1,
                            'DeltaAvg': delta_avg, 'DeltaLag1': delta_lag1,
                            'IsCloser': is_closer, 'IsLeader': is_leader,
                            'WeightChange': weight_change,
                            'Closing_x_Trainer': closing_x_trainer, 'Age_x_Beyer': age_x_beyer,
                            # V11 NEW: Enhanced Trainer Stats (4)
                            'TrainerWinRate_30d': trainer_win_rate_30d,
                            'TrainerStarts_30d': trainer_starts_30d,
                            'TrainerWinRate_Track': trainer_win_rate_track,
                            'TrainerWinRate_Dist': trainer_win_rate_dist,
                            # V11 NEW: Derived Features (7)
                            'DaysSinceRace': days_since_race,
                            'IsFresh': is_fresh,
                            'IsBackedUp': is_backed_up,
                            'IsSpelled': is_spelled,
                            'ConsistencyScore': consistency_score,
                            'RecentTrend': recent_trend,
                            'BoxType': box_type,
                            # V11 NEW: Interactions (8)
                            'TimeBest_x_TrainerForm': time_best_x_trainer_form,
                            'WinRate_x_BoxType': win_rate_x_box_type,
                            'Dist_x_Track_Specialist': dist_x_track_specialist,
                            'Beyer_x_PeakAge': beyer_x_peak_age,
                            'Closing_x_Box': closing_x_box,
                            'TrainerForm_x_DogForm': trainer_form_x_dog_form,
                            'Consistency_x_WinRate': consistency_x_win_rate,
                            'Trend_x_Trainer': trend_x_trainer
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
                'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
                'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
                'weight': r['Weight'],
                'box': r['Box']
            })
            
            # Update trainer stats
            if pd.notna(r['TrainerID']):
                trainer_id = r['TrainerID']
                won = r['Won']
                
                # All-time
                trainer_all_time[trainer_id]['runs'] += 1
                if won == 1:
                    trainer_all_time[trainer_id]['wins'] += 1
                
                # Recent (for 30-day lookback)
                trainer_recent[trainer_id].append((race_date, won, track_id, distance))
                # Keep only last 90 days of data
                cutoff = race_date - timedelta(days=90)
                trainer_recent[trainer_id] = [(d, w, tid, dist) for d, w, tid, dist in trainer_recent[trainer_id] if d >= cutoff]
                
                # By track
                trainer_track[trainer_id][track_id]['runs'] += 1
                if won == 1:
                    trainer_track[trainer_id][track_id]['wins'] += 1
                
                # By distance (rounded to 100m)
                dist_key = round(distance / 100) * 100
                trainer_distance[trainer_id][dist_key]['runs'] += 1
                if won == 1:
                    trainer_distance[trainer_id][dist_key]['wins'] += 1
            
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
        # V5 Base (19)
        'TimeBest', 'TimeAvg3', 'TimeLag1', 'PosAvg', 'WinRate',
        'TrainerWinRate', 'DistWinRate', 'TrackWinRate', 'BloodlineScore',
        'CareerStarts', 'Box', 'BeyerBest', 'BeyerAvg', 'BeyerLag1',
        'TimeBest_x_Trainer', 'Beyer_x_Trainer', 'TimeBest_x_DistExp',
        'WinRate_x_Dist', 'WinRate_x_Track',
        # V5 Age/Closing (15)
        'AgeMonths', 'IsPeakAge', 'IsYoung', 'IsVeteran', 'Age_x_Experience',
        'ClosingAvg', 'ClosingBest', 'ClosingLag1',
        'DeltaAvg', 'DeltaLag1', 'IsCloser', 'IsLeader',
        'WeightChange', 'Closing_x_Trainer', 'Age_x_Beyer',
        # V11 Enhanced Trainer (4)
        'TrainerWinRate_30d', 'TrainerStarts_30d',
        'TrainerWinRate_Track', 'TrainerWinRate_Dist',
        # V11 Derived (7)
        'DaysSinceRace', 'IsFresh', 'IsBackedUp', 'IsSpelled',
        'ConsistencyScore', 'RecentTrend', 'BoxType',
        # V11 Interactions (8)
        'TimeBest_x_TrainerForm', 'WinRate_x_BoxType',
        'Dist_x_Track_Specialist', 'Beyer_x_PeakAge',
        'Closing_x_Box', 'TrainerForm_x_DogForm',
        'Consistency_x_WinRate', 'Trend_x_Trainer'
    ]
    
    print(f"Feature count: {len(feature_cols)} (V5: 34, V11 new: {len(feature_cols) - 34})")
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['Won']
    
    print("\n[6/7] Training with GPU (500 random combinations)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Parameter distributions for random search
    param_dist = {
        'n_estimators': randint(50, 400),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.19),
        'min_child_samples': randint(20, 250),
        'subsample': uniform(0.6, 0.35),
        'colsample_bytree': uniform(0.6, 0.35),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0, 1.0),
        'num_leaves': randint(15, 64)
    }
    
    n_iter = 500  # Faster than V5 GPU search
    print(f"  Random combinations: {n_iter}")
    print(f"  Total fits (3-fold CV): {n_iter * 3}")
    
    lgb = LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=42,
        verbose=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb, param_dist, n_iter=n_iter, cv=3,
        scoring='roc_auc', n_jobs=1, verbose=2, random_state=42
    )
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest params: {random_search.best_params_}")
    print(f"Best CV AUC: {random_search.best_score_:.4f}")
    
    print("\n[7/7] Evaluating...")
    
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print(f"\nTest AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nML Top Pick:")
    print(f"  Total races: {len(race_leaders):,}")
    print(f"  Wins: {race_leaders['Won'].sum():,}")
    print(f"  Strike Rate: {race_leaders['Won'].mean()*100:.1f}%")
    
    print("\nTop 20 Feature Importance:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.head(20).to_string(index=False))
    
    # Highlight V11 new features
    v11_features = ['TrainerWinRate_30d', 'TrainerStarts_30d', 'TrainerWinRate_Track', 
                    'TrainerWinRate_Dist', 'DaysSinceRace', 'IsFresh', 'IsBackedUp', 
                    'IsSpelled', 'ConsistencyScore', 'RecentTrend', 'BoxType',
                    'TimeBest_x_TrainerForm', 'WinRate_x_BoxType', 'Dist_x_Track_Specialist',
                    'Beyer_x_PeakAge', 'Closing_x_Box', 'TrainerForm_x_DogForm',
                    'Consistency_x_WinRate', 'Trend_x_Trainer']
    v11_importance = importance[importance['Feature'].isin(v11_features)]
    print("\n--- V11 NEW FEATURES IMPORTANCE ---")
    print(v11_importance.to_string(index=False))
    
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
        'best_params': random_search.best_params_
    }
    with open('models/pace_v11_enhanced.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved to models/pace_v11_enhanced.pkl")
    print("="*70)

if __name__ == "__main__":
    train_v11_model()
