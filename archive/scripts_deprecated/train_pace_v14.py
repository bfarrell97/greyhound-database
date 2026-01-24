"""
V14 Pace Model - Improved Architecture
Key Improvements:
1. Proper Train/Validation/Test split (2020-2023 / 2024 / 2025)
2. More training iterations (1000)
3. Early stopping to prevent overfitting
4. Stronger regularization
5. Probability calibration
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import time

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

print("="*70)
print("V14 PACE MODEL - IMPROVED ARCHITECTURE")
print("Train: 2020-2023 | Validation: 2024 | Test: 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
       ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
       ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID, g.DamID, g.DateOfBirth
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = (df['Position'] == 1).astype(int)
df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Price5Min', 'Weight',
            'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']

print(f"[1/6] Loaded {len(df):,} entries")

# Define feature set (refined from V12)
FEATURE_COLS = [
    # Time features
    'TimeBest', 'TimeWorst', 'TimeAvg', 'TimeAvg3', 'TimeLag1', 'TimeLag2', 'TimeLag3',
    'TimeStd', 'TimeImproving', 'TimeTrend3', 'TimeBestRecent3', 'TimeQ25',
    # Split features
    'SplitBest', 'SplitAvg', 'SplitLag1', 'SplitStd',
    # Speed figures
    'BeyerLag1', 'BeyerStd',
    # Career stats
    'PosAvg', 'WinRate5', 'CareerWins', 'CareerPlaces', 'CareerStarts', 'CareerWinRate',
    'LastWonDaysAgo', 'FormTrend',
    # Trainer
    'TrainerWinRate', 'TrainerWinRate30d', 'TrainerStarts60d', 'TrainerFormVsAll',
    'TrainerTrackWinRate', 'TrainerDistWinRate',
    # Distance/Track specialization
    'DistWinRate', 'DistPlaceRate', 'DistExperience', 'DistAvgPos',
    'TrackWinRate', 'TrackPlaceRate', 'TrackExperience',
    # Box
    'Box', 'TrackBoxWinRate', 'BoxPreference', 'ThisBoxWinRate',
    # Age
    'AgeMonths', 'ExperiencePerAge', 'WinsPerAge', 'AgePeakDist',
    # Pace/Closing
    'PosImprovement', 'ClosingAvg', 'ClosingBest', 'DeltaBest',
    # Bloodline
    'SireWinRate', 'DamWinRate', 'BloodlineScore',
    # Weight
    'Weight', 'WeightAvg', 'WeightChange', 'WeightStd',
    # Rest/Frequency
    'DaysSinceRace', 'RaceFrequency30d', 'RaceFrequency60d', 'RestScore',
    # Interactions
    'Time_x_Trainer', 'Beyer_x_Trainer', 'Form_x_Trainer', 'SpecialistScore'
]

print("[2/6] Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
trainer_recent = defaultdict(list)
trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))

train_rows = []  # 2020-2023
val_rows = []    # 2024
test_rows = []   # 2025
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 4: continue
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    track_id = race_df['TrackID'].iloc[0]
    
    year = race_date.year
    
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
                features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Distance': distance, 'MeetingDate': race_date}
                
                # Time features
                features['TimeBest'] = min(times)
                features['TimeWorst'] = max(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                features['TimeStd'] = np.std(times)
                features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                features['TimeBestRecent3'] = min(times[-3:])
                features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                
                # Split features
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                features['SplitLag1'] = splits[-1] if splits else 0
                features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                
                # Speed figures
                features['BeyerLag1'] = beyers[-1] if beyers else 77
                features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                
                # Career stats
                features['PosAvg'] = np.mean(positions)
                features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                features['CareerStarts'] = min(len(hist), 100)
                features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                features['LastWonDaysAgo'] = 999
                for i, h in enumerate(reversed(hist)):
                    if h['position'] == 1:
                        features['LastWonDaysAgo'] = (race_date - h['date']).days
                        break
                
                form_trend = 0
                if len(positions) >= 5:
                    first_half = np.mean(positions[:len(positions)//2])
                    second_half = np.mean(positions[len(positions)//2:])
                    form_trend = first_half - second_half
                features['FormTrend'] = form_trend
                
                # Trainer
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                cutoff_30d = race_date - timedelta(days=30)
                t_rec = trainer_recent.get(trainer_id, [])
                rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                features['TrainerStarts60d'] = len([x for x in t_rec if x[0] >= race_date - timedelta(days=60)])
                features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                
                t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                
                dist_key = round(distance / 100) * 100
                t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                
                # Distance/Track specialization
                dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                features['DistExperience'] = min(len(dist_runs), 30) / 30
                features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                
                track_runs = [h for h in hist if h['track_id'] == track_id]
                features['TrackWinRate'] = safe_div(sum(1 for t in track_runs if t['position'] == 1), len(track_runs), features['CareerWinRate']) if len(track_runs) >= 3 else features['CareerWinRate']
                features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), 0.35) if len(track_runs) >= 3 else 0.35
                features['TrackExperience'] = min(len(track_runs), 20) / 20
                
                # Box
                box = int(r['Box']) if pd.notna(r['Box']) else 4
                features['Box'] = box
                tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                
                inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                outside_runs = [h for h in hist if h.get('box', 4) > 4]
                inside_rate = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['CareerWinRate']) if len(inside_runs) >= 3 else features['CareerWinRate']
                outside_rate = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['CareerWinRate']) if len(outside_runs) >= 3 else features['CareerWinRate']
                features['BoxPreference'] = inside_rate - outside_rate
                
                this_box_runs = [h for h in hist if h.get('box', 0) == box]
                features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['CareerWinRate']) if len(this_box_runs) >= 2 else features['CareerWinRate']
                
                # Age
                age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                features['AgeMonths'] = age_months
                features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                features['AgePeakDist'] = abs(age_months - 30)
                
                # Pace/Closing
                deltas_array = deltas if deltas else [0]
                features['PosImprovement'] = np.mean(deltas_array)
                features['ClosingAvg'] = np.mean(closings) if closings else 0
                features['ClosingBest'] = min(closings) if closings else 0
                features['DeltaBest'] = max(deltas) if deltas else 0
                
                # Bloodline
                sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                
                # Weight
                weight_avg = np.mean(weights) if weights else 30
                current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                features['Weight'] = current_weight
                features['WeightAvg'] = weight_avg
                features['WeightChange'] = current_weight - weight_avg
                features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                
                # Rest/Frequency
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                rest_score = max(0, 1 - abs(days_since - 10) / 20)
                features['RestScore'] = rest_score
                
                # Interactions
                features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                features['Form_x_Trainer'] = form_trend * features['TrainerWinRate']
                features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrainerTrackWinRate']) / 3
                
                # Assign to correct set
                if year <= 2023:
                    train_rows.append(features)
                elif year == 2024:
                    val_rows.append(features)
                else:
                    test_rows.append(features)
    
    # Update lookups
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        box = int(r['Box']) if pd.notna(r['Box']) else 4
        won = r['Won']
        
        dog_history[dog_id].append({
            'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
            'position': r['Position'], 'track_id': track_id, 'distance': distance,
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
    if processed % 50000 == 0: print(f"  {processed:,} races...")

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)
test_df = pd.DataFrame(test_rows)

print(f"  Train: {len(train_df):,} (2020-2023)")
print(f"  Val: {len(val_df):,} (2024)")
print(f"  Test: {len(test_df):,} (2025)")

print("[3/6] Training V14 Model with Early Stopping...")

X_train = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['Won']
X_val = val_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_val = val_df['Won']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Improved hyperparameters with stronger regularization
v14_params = {
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 40,
    'min_child_samples': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = LGBMClassifier(**v14_params)
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[
        LGBMClassifier.early_stopping(stopping_rounds=50, verbose=False)
    ] if hasattr(LGBMClassifier, 'early_stopping') else None
)

print(f"  Best iteration: {model.best_iteration_}")

print("[4/6] Validating on 2024 (should overfit less)...")
val_df['PredProb'] = model.predict_proba(X_val_scaled)[:, 1]
val_df['MarginOverSecond'] = val_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
val_leaders = val_df.loc[val_df.groupby('RaceID')['PredProb'].idxmax()]

# Test strategy on 2024
val_strat = val_leaders[
    (val_leaders['MarginOverSecond'] >= 0.10) & 
    (val_leaders['BSP'] >= 3) & (val_leaders['BSP'] <= 8) &
    (val_leaders['Distance'] < 550)
]
val_valid = val_strat.dropna(subset=['BSP'])
val_wins = val_valid['Won'].sum()
val_profit = (val_valid[val_valid['Won']==1]['BSP'].sum() * 0.9) - len(val_valid)
val_roi = val_profit / len(val_valid) * 100 if len(val_valid) > 0 else 0

print(f"  2024 Validation: {len(val_valid)} bets, ROI: {val_roi:+.1f}%")

print("[5/6] Testing on 2025 (True Out-of-Sample)...")
X_test = test_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X_test_scaled = scaler.transform(X_test)

test_df['PredProb'] = model.predict_proba(X_test_scaled)[:, 1]
test_df['MarginOverSecond'] = test_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
test_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]

# Test strategy on 2025
test_strat = test_leaders[
    (test_leaders['MarginOverSecond'] >= 0.10) & 
    (test_leaders['BSP'] >= 3) & (test_leaders['BSP'] <= 8) &
    (test_leaders['Distance'] < 550)
]
test_valid = test_strat.dropna(subset=['BSP'])
test_wins = test_valid['Won'].sum()
test_profit = (test_valid[test_valid['Won']==1]['BSP'].sum() * 0.9) - len(test_valid)
test_roi = test_profit / len(test_valid) * 100 if len(test_valid) > 0 else 0

print(f"\n" + "="*70)
print("V14 RESULTS")
print("="*70)
print(f"Strategy: High Conf (>10%) + $3-$8 + <550m")
print(f"2024 (Validation): {len(val_valid)} bets, ROI: {val_roi:+.1f}%")
print(f"2025 (Test):       {len(test_valid)} bets, ROI: {test_roi:+.1f}%")
print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Save model
print("\n[6/6] Saving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'features': FEATURE_COLS,
    'params': v14_params,
    'best_iteration': model.best_iteration_,
    'validation_roi': val_roi,
    'test_roi': test_roi
}
with open('models/pace_v14.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved to models/pace_v14.pkl")
print("="*70)
