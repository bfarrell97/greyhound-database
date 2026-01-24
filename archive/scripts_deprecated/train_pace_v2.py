"""
Enhanced Pace Model V2 - Comprehensive Feature Engineering
===========================================================
All creative features:
1. Early Speed Index
2. Grade Momentum
3. Trainer Form
4. Sire/Dam Win Rate
5. Box Performance
6. Weight Change
7. Career Stage
8. Prize Money Trend
9. Track Specialist
10. Distance Specialist
11. Competitor Strength
12. Form Consistency
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

def grade_to_numeric(grade):
    """Convert grade to numeric value. Higher = better class"""
    if pd.isna(grade) or grade == '':
        return 4  # Default middle
    grade = str(grade).upper().strip()
    if 'MAIDEN' in grade:
        return 1
    try:
        g = int(grade)
        return 9 - g  # Grade 1 = 8, Grade 7 = 2
    except:
        return 4

def train_enhanced_model():
    print("="*70)
    print("ENHANCED PACE MODEL V2 - ALL CREATIVE FEATURES")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load benchmarks
    print("\n[1/8] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    # Load ALL data including new columns
    print("\n[2/8] Loading comprehensive race data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, ge.EarlySpeed, ge.Rating, ge.Weight,
           ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
           ge.IncomingGrade, ge.OutgoingGrade, ge.PrizeMoney,
           ge.TrainerID, ge.Margin,
           r.Distance, r.Grade as RaceGrade, rm.MeetingDate, t.TrackName, rm.TrackID,
           g.Sire, g.Dam, g.SireID, g.DamID, g.DateOfBirth, g.Sex, g.Starts as CareerStarts
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
    
    # Convert types
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    df['Placed'] = (df['Position'] <= 3).astype(int)
    
    for col in ['FinishTime', 'Split', 'EarlySpeed', 'Rating', 'Weight', 
                'FirstSplitPosition', 'SecondSplitTime', 'BSP', 'PrizeMoney', 'Margin']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    df['IncomingGradeNum'] = df['IncomingGrade'].apply(grade_to_numeric)
    df['OutgoingGradeNum'] = df['OutgoingGrade'].apply(grade_to_numeric)
    
    # Calculate age
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    
    print(f"Loaded {len(df):,} entries with {len(df.columns)} columns")
    
    # [3/8] Build historical lookup tables
    print("\n[3/8] Building historical lookups...")
    
    # Dog history: times, positions, boxes, tracks, distances, weights, grades, prizes
    dog_history = defaultdict(list)
    
    # Trainer win rate tracking
    trainer_recent = defaultdict(lambda: {'wins': 0, 'runs': 0})
    
    # Sire/Dam tracking
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    
    # [4/8] Build features
    print("\n[4/8] Building comprehensive features...")
    
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        # Get competitor ratings for this race (for strength calculation)
        race_ratings = race_df['Rating'].dropna()
        avg_race_rating = race_ratings.mean() if len(race_ratings) > 0 else 0
        
        # Only create features for TEST period (2024+)
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                hist = dog_history.get(dog_id, [])
                
                if len(hist) >= 3:
                    # Filter valid times
                    recent = hist[-10:]  # Last 10 races
                    recent_times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    recent_splits = [h['split'] for h in recent if h['split'] is not None]
                    recent_pos = [h['position'] for h in recent if h['position'] is not None]
                    
                    if len(recent_times) >= 3:
                        # === ORIGINAL TIME FEATURES ===
                        time_lag1 = recent_times[-1]
                        time_lag2 = recent_times[-2] if len(recent_times) >= 2 else time_lag1
                        time_lag3 = recent_times[-3] if len(recent_times) >= 3 else time_lag2
                        time_avg3 = np.mean(recent_times[-3:])
                        time_avg5 = np.mean(recent_times[-5:]) if len(recent_times) >= 5 else time_avg3
                        time_std = np.std(recent_times) if len(recent_times) >= 3 else 0
                        time_trend = time_lag1 - time_lag3
                        time_best = min(recent_times)
                        
                        # Split features
                        split_avg = np.mean(recent_splits) if recent_splits else 0
                        split_best = min(recent_splits) if recent_splits else 0
                        
                        # Position features
                        pos_avg = np.mean(recent_pos) if recent_pos else 4
                        win_rate = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                        place_rate = sum(1 for p in recent_pos if p <= 3) / len(recent_pos) if recent_pos else 0
                        
                        # === 1. EARLY SPEED INDEX ===
                        early_speeds = [h['early_speed'] for h in recent if h['early_speed'] is not None]
                        first_splits = [h['first_split_pos'] for h in recent if h['first_split_pos'] is not None]
                        early_speed_avg = np.mean(early_speeds) if early_speeds else 0
                        first_split_avg = np.mean(first_splits) if first_splits else 4
                        early_speed_index = (early_speed_avg * 10) - first_split_avg  # Higher = faster start
                        
                        # === 2. GRADE MOMENTUM ===
                        grades = [h['incoming_grade'] for h in recent]
                        if len(grades) >= 2:
                            grade_momentum = grades[-1] - grades[0]  # Positive = moving up
                        else:
                            grade_momentum = 0
                        current_grade = grades[-1] if grades else 4
                        
                        # === 3. TRAINER FORM ===
                        trainer_id = r['TrainerID']
                        trainer_data = trainer_recent.get(trainer_id, {'wins': 0, 'runs': 0})
                        trainer_win_rate = trainer_data['wins'] / trainer_data['runs'] if trainer_data['runs'] > 20 else 0.12
                        
                        # === 4. SIRE/DAM WIN RATE ===
                        sire_id = r['SireID']
                        dam_id = r['DamID']
                        sire_data = sire_stats.get(sire_id, {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(dam_id, {'wins': 0, 'runs': 0})
                        sire_win_rate = sire_data['wins'] / sire_data['runs'] if sire_data['runs'] > 50 else 0.12
                        dam_win_rate = dam_data['wins'] / dam_data['runs'] if dam_data['runs'] > 30 else 0.12
                        
                        # === 5. BOX PERFORMANCE ===
                        current_box = r['Box']
                        box_runs = [h for h in recent if h['box'] == current_box]
                        box_win_rate = sum(1 for b in box_runs if b['position'] == 1) / len(box_runs) if len(box_runs) >= 2 else win_rate
                        
                        # === 6. WEIGHT CHANGE ===
                        weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
                        avg_weight = np.mean(weights) if weights else 30
                        current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else avg_weight
                        weight_change = current_weight - avg_weight
                        
                        # === 7. CAREER STAGE ===
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        career_starts = len(hist)
                        is_experienced = 1 if career_starts > 20 else 0
                        is_young = 1 if age_months < 24 else 0
                        
                        # === 8. PRIZE MONEY TREND ===
                        prizes = [h['prize'] for h in recent if h['prize'] is not None and h['prize'] > 0]
                        recent_prize_avg = np.mean(prizes[-5:]) if len(prizes) >= 5 else (np.mean(prizes) if prizes else 0)
                        career_prize_avg = np.mean(prizes) if prizes else 0
                        prize_trend = recent_prize_avg - career_prize_avg
                        
                        # === 9. TRACK SPECIALIST ===
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        track_win_rate = sum(1 for t in track_runs if t['position'] == 1) / len(track_runs) if len(track_runs) >= 3 else win_rate
                        track_experience = min(len(track_runs), 20) / 20  # 0-1 scale
                        
                        # === 10. DISTANCE SPECIALIST ===
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        dist_win_rate = sum(1 for d in dist_runs if d['position'] == 1) / len(dist_runs) if len(dist_runs) >= 3 else win_rate
                        dist_experience = min(len(dist_runs), 20) / 20
                        
                        # === 11. COMPETITOR STRENGTH ===
                        dog_rating = r['Rating'] if pd.notna(r['Rating']) else 0
                        rating_vs_field = dog_rating - avg_race_rating if avg_race_rating > 0 else 0
                        
                        # === 12. FORM CONSISTENCY ===
                        pos_std = np.std(recent_pos) if len(recent_pos) >= 3 else 3
                        time_consistency = 1 / (1 + time_std)  # Higher = more consistent
                        
                        # Days since last race
                        last_race_date = hist[-1]['date']
                        days_since = min((race_date - last_race_date).days, 90)
                        
                        feature_rows.append({
                            'RaceID': race_id,
                            'Won': r['Won'],
                            'BSP': r['BSP'],
                            'Tier': tier,
                            # Original features
                            'TimeLag1': time_lag1,
                            'TimeLag2': time_lag2,
                            'TimeAvg3': time_avg3,
                            'TimeAvg5': time_avg5,
                            'TimeStd': time_std,
                            'TimeTrend': time_trend,
                            'TimeBest': time_best,
                            'SplitAvg': split_avg,
                            'SplitBest': split_best,
                            'PosAvg': pos_avg,
                            'WinRate': win_rate,
                            'PlaceRate': place_rate,
                            'DaysSince': days_since,
                            'Box': r['Box'],
                            'Distance': distance,
                            # NEW FEATURES
                            'EarlySpeedIndex': early_speed_index,
                            'FirstSplitAvg': first_split_avg,
                            'GradeMomentum': grade_momentum,
                            'CurrentGrade': current_grade,
                            'TrainerWinRate': trainer_win_rate,
                            'SireWinRate': sire_win_rate,
                            'DamWinRate': dam_win_rate,
                            'BoxWinRate': box_win_rate,
                            'WeightChange': weight_change,
                            'AgeMonths': age_months,
                            'CareerStarts': min(career_starts, 100),
                            'IsExperienced': is_experienced,
                            'IsYoung': is_young,
                            'PrizeTrend': prize_trend,
                            'TrackWinRate': track_win_rate,
                            'TrackExperience': track_experience,
                            'DistWinRate': dist_win_rate,
                            'DistExperience': dist_experience,
                            'RatingVsField': rating_vs_field,
                            'PosStd': pos_std,
                            'TimeConsistency': time_consistency
                        })
        
        # Update histories for ALL rows
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            dog_history[dog_id].append({
                'date': race_date,
                'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'split': r['Split'] if pd.notna(r['Split']) else None,
                'position': r['Position'],
                'box': r['Box'],
                'weight': r['Weight'],
                'track_id': track_id,
                'distance': distance,
                'early_speed': r['EarlySpeed'],
                'first_split_pos': r['FirstSplitPosition'],
                'incoming_grade': r['IncomingGradeNum'],
                'prize': r['PrizeMoney']
            })
            
            # Update trainer stats
            trainer_id = r['TrainerID']
            if pd.notna(trainer_id):
                trainer_recent[trainer_id]['runs'] += 1
                if r['Won'] == 1:
                    trainer_recent[trainer_id]['wins'] += 1
            
            # Update sire/dam stats
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
    
    # [5/8] Prepare data
    print("\n[5/8] Preparing train/test split...")
    
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    feat_df = feat_df[(feat_df['TimeAvg5'] > -5) & (feat_df['TimeAvg5'] < 5)]
    
    # Use 2024 Jan-Jun for train, rest for test (or subsample)
    train_df = feat_df[feat_df['RaceID'] % 3 != 0]  # 2/3 for training
    test_df = feat_df[feat_df['RaceID'] % 3 == 0]   # 1/3 for testing
    
    if len(train_df) > 300000:
        print(f"  Subsampling from {len(train_df):,} to 300,000...")
        train_df = train_df.sample(n=300000, random_state=42)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    # Feature columns
    feature_cols = [
        'TimeLag1', 'TimeLag2', 'TimeAvg3', 'TimeAvg5', 'TimeStd', 'TimeTrend', 'TimeBest',
        'SplitAvg', 'SplitBest', 'PosAvg', 'WinRate', 'PlaceRate', 'DaysSince', 'Box', 'Distance',
        'EarlySpeedIndex', 'FirstSplitAvg', 'GradeMomentum', 'CurrentGrade',
        'TrainerWinRate', 'SireWinRate', 'DamWinRate', 'BoxWinRate',
        'WeightChange', 'AgeMonths', 'CareerStarts', 'IsExperienced', 'IsYoung',
        'PrizeTrend', 'TrackWinRate', 'TrackExperience', 'DistWinRate', 'DistExperience',
        'RatingVsField', 'PosStd', 'TimeConsistency'
    ]
    
    print(f"Feature count: {len(feature_cols)}")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Won']
    
    # [6/8] Scale and train
    print("\n[6/8] Training with GridSearchCV...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 7],
        'learning_rate': [0.1],
        'min_samples_split': [100],
        'subsample': [0.8]
    }
    
    print(f"Grid: {param_grid}")
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    # [7/8] Evaluate
    print("\n[7/8] Evaluating...")
    
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
    
    # [8/8] Backtest
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
    
    # Save model
    print("\n" + "="*70)
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_params': grid_search.best_params_
    }
    with open('models/pace_v2_enhanced.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    print("Saved to models/pace_v2_enhanced.pkl")
    print("="*70)

if __name__ == "__main__":
    train_enhanced_model()
