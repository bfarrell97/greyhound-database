"""
V15 Pace Model - PIR Features + More Combinations
Key Improvements:
1. PIR (Position In Running) features from InRun column
2. More feature combinations
3. Log transforms for skewed features
4. Rank-within-race features
5. Target: 5%+ ROI out-of-sample
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import time

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def parse_inrun(inrun_str):
    """Parse InRun string (e.g., '123') to list of positions [1, 2, 3]"""
    if pd.isna(inrun_str) or not str(inrun_str).strip():
        return None
    try:
        return [int(c) for c in str(inrun_str).strip() if c.isdigit()]
    except:
        return None

print("="*70)
print("V15 PACE MODEL - PIR FEATURES + MORE COMBINATIONS")
print("Train: 2020-2023 | Validation: 2024 | Test: 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
       ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
       ge.InRun, ge.Margin,
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
df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce')

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
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

# Parse InRun
df['InRunParsed'] = df['InRun'].apply(parse_inrun)

print(f"[1/6] Loaded {len(df):,} entries")
pir_count = df['InRunParsed'].notna().sum()
print(f"       PIR data available: {pir_count:,} ({pir_count/len(df)*100:.1f}%)")

print("[2/6] Building features with PIR...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
trainer_recent = defaultdict(list)
sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))

train_rows = []
val_rows = []
test_rows = []
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
            margins = [h['margin'] for h in recent if h['margin'] is not None]
            
            # PIR data from history
            pir_first = [h['pir'][0] for h in recent if h.get('pir') and len(h['pir']) >= 1]
            pir_last = [h['pir'][-1] for h in recent if h.get('pir') and len(h['pir']) >= 1]
            pir_mid = [h['pir'][len(h['pir'])//2] for h in recent if h.get('pir') and len(h['pir']) >= 2]
            
            if len(times) >= 3:
                features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Distance': distance, 'MeetingDate': race_date}
                
                # ============ CORE TIME FEATURES ============
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
                
                # ============ SPLIT FEATURES ============
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                features['SplitLag1'] = splits[-1] if splits else 0
                features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                
                # ============ PIR FEATURES (NEW) ============
                if pir_first:
                    features['PIRFirstAvg'] = np.mean(pir_first)  # Avg early position
                    features['PIRFirstBest'] = min(pir_first)     # Best early position
                    features['PIRFirstLag1'] = pir_first[-1]      # Last race early position
                    features['LeaderRatio'] = sum(1 for p in pir_first if p == 1) / len(pir_first)  # % times led early
                else:
                    features['PIRFirstAvg'] = 4
                    features['PIRFirstBest'] = 4
                    features['PIRFirstLag1'] = 4
                    features['LeaderRatio'] = 0.125
                
                if pir_last:
                    features['PIRLastAvg'] = np.mean(pir_last)    # Avg finishing position from PIR
                    features['PIRLastBest'] = min(pir_last)       # Best finishing position
                    features['PIRLastLag1'] = pir_last[-1]        # Last race finish
                else:
                    features['PIRLastAvg'] = 4
                    features['PIRLastBest'] = 4
                    features['PIRLastLag1'] = 4
                
                # PIR Movement (early vs late)
                if pir_first and pir_last:
                    pir_moves = [pir_first[i] - pir_last[i] for i in range(min(len(pir_first), len(pir_last)))]
                    features['PIRMovement'] = np.mean(pir_moves)  # Positive = improves through race
                    features['PIRMoveStd'] = np.std(pir_moves) if len(pir_moves) >= 3 else 0
                    features['CloserRatio'] = sum(1 for m in pir_moves if m > 0) / len(pir_moves)  # % times closed
                else:
                    features['PIRMovement'] = 0
                    features['PIRMoveStd'] = 0
                    features['CloserRatio'] = 0.5
                
                # ============ MARGIN FEATURES (NEW) ============
                if margins:
                    features['MarginAvg'] = np.mean(margins)
                    features['MarginBest'] = min(margins)  # Closest margin
                    features['MarginLag1'] = margins[-1]
                else:
                    features['MarginAvg'] = 3
                    features['MarginBest'] = 0
                    features['MarginLag1'] = 3
                
                # ============ SPEED & CAREER ============
                features['BeyerLag1'] = beyers[-1] if beyers else 77
                features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                features['BeyerBest'] = max(beyers) if beyers else 77
                features['BeyerAvg'] = np.mean(beyers) if beyers else 77
                
                features['PosAvg'] = np.mean(positions)
                features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                features['PlaceRate5'] = sum(1 for p in positions[-5:] if p <= 3) / min(5, len(positions))
                features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                features['CareerStarts'] = min(len(hist), 100)
                features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                features['CareerPlaceRate'] = safe_div(features['CareerPlaces'], features['CareerStarts'], 0.35)
                
                # Log transform for skewed features
                features['LogCareerStarts'] = np.log1p(features['CareerStarts'])
                
                features['LastWonDaysAgo'] = 999
                for i, h in enumerate(reversed(hist)):
                    if h['position'] == 1:
                        features['LastWonDaysAgo'] = (race_date - h['date']).days
                        break
                features['LogLastWonDaysAgo'] = np.log1p(features['LastWonDaysAgo'])
                
                form_trend = 0
                if len(positions) >= 5:
                    first_half = np.mean(positions[:len(positions)//2])
                    second_half = np.mean(positions[len(positions)//2:])
                    form_trend = first_half - second_half
                features['FormTrend'] = form_trend
                
                # ============ TRAINER ============
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                cutoff_30d = race_date - timedelta(days=30)
                t_rec = trainer_recent.get(trainer_id, [])
                rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                features['TrainerStarts60d'] = len([x for x in t_rec if x[0] >= race_date - timedelta(days=60)])
                features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                
                # ============ SPECIALIZATION ============
                dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                features['DistExperience'] = min(len(dist_runs), 30) / 30
                features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                
                track_runs = [h for h in hist if h['track_id'] == track_id]
                features['TrackWinRate'] = safe_div(sum(1 for t in track_runs if t['position'] == 1), len(track_runs), features['CareerWinRate']) if len(track_runs) >= 3 else features['CareerWinRate']
                features['TrackExperience'] = min(len(track_runs), 20) / 20
                
                # ============ BOX ============
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
                
                # ============ AGE ============
                age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                features['AgeMonths'] = age_months
                features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                features['AgePeakDist'] = abs(age_months - 30)
                
                # ============ PACE/CLOSING ============
                deltas_array = deltas if deltas else [0]
                features['PosImprovement'] = np.mean(deltas_array)
                features['ClosingAvg'] = np.mean(closings) if closings else 0
                features['ClosingBest'] = min(closings) if closings else 0
                features['DeltaBest'] = max(deltas) if deltas else 0
                
                # ============ BLOODLINE ============
                sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                
                # ============ WEIGHT ============
                weight_avg = np.mean(weights) if weights else 30
                current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                features['Weight'] = current_weight
                features['WeightAvg'] = weight_avg
                features['WeightChange'] = current_weight - weight_avg
                features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                
                # ============ REST ============
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                features['LogDaysSince'] = np.log1p(days_since)
                features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                rest_score = max(0, 1 - abs(days_since - 10) / 20)
                features['RestScore'] = rest_score
                
                # ============ INTERACTIONS ============
                features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                features['Form_x_Trainer'] = form_trend * features['TrainerWinRate']
                features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrackWinRate']) / 3
                
                # PIR x Other
                features['PIR_x_Beyer'] = features['PIRFirstAvg'] * features['BeyerAvg']
                features['PIR_x_Career'] = features['LeaderRatio'] * features['CareerWinRate']
                features['PIR_x_Trainer'] = features['LeaderRatio'] * features['TrainerWinRate']
                features['Closing_x_PIR'] = features['CloserRatio'] * features['ClosingAvg']
                
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
            'weight': r['Weight'], 'box': box, 'split': r['Split'] if pd.notna(r['Split']) else None,
            'margin': r['Margin'] if pd.notna(r['Margin']) else None,
            'pir': r['InRunParsed']
        })
        
        if pd.notna(r['TrainerID']):
            tid = r['TrainerID']
            trainer_all[tid]['runs'] += 1
            if won: trainer_all[tid]['wins'] += 1
            trainer_recent[tid].append((race_date, won, track_id, distance))
            cutoff = race_date - timedelta(days=120)
            trainer_recent[tid] = [x for x in trainer_recent[tid] if x[0] >= cutoff]
        
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

# Get feature columns (all numeric columns except metadata)
exclude_cols = ['RaceID', 'Won', 'BSP', 'Distance', 'MeetingDate']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} (2020-2023)")
print(f"  Val: {len(val_df):,} (2024)")
print(f"  Test: {len(test_df):,} (2025)")
print(f"  Features: {len(FEATURE_COLS)}")

# Correlation filtering
print("[3/7] Removing highly correlated features (>0.9)...")
X_corr = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
corr_matrix = X_corr.corr().abs()

# Find pairs with correlation > 0.9
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

if to_drop:
    print(f"  Dropping {len(to_drop)} highly correlated features:")
    for f in to_drop[:10]:  # Show first 10
        print(f"    - {f}")
    if len(to_drop) > 10:
        print(f"    ... and {len(to_drop) - 10} more")
    
    FEATURE_COLS = [c for c in FEATURE_COLS if c not in to_drop]
    print(f"  Remaining features: {len(FEATURE_COLS)}")
else:
    print("  No highly correlated features found")

print("[4/7] Training V15 Model...")

X_train = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['Won']
X_val = val_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_val = val_df['Won']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model with moderate regularization
v15_params = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 7,
    'num_leaves': 50,
    'min_child_samples': 80,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

model = LGBMClassifier(**v15_params)
model.fit(X_train_scaled, y_train)

print("[5/7] Validating on 2024...")
val_df['PredProb'] = model.predict_proba(X_val_scaled)[:, 1]
val_df['MarginOverSecond'] = val_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
val_leaders = val_df.loc[val_df.groupby('RaceID')['PredProb'].idxmax()]

# Test all confidence levels
print("\n2024 Validation Results by Confidence:")
for conf in [0.05, 0.08, 0.10, 0.12, 0.15]:
    val_strat = val_leaders[
        (val_leaders['MarginOverSecond'] >= conf) & 
        (val_leaders['BSP'] >= 3) & (val_leaders['BSP'] <= 8)
    ]
    val_valid = val_strat.dropna(subset=['BSP'])
    if len(val_valid) > 0:
        val_profit = (val_valid[val_valid['Won']==1]['BSP'].sum() * 0.9) - len(val_valid)
        val_roi = val_profit / len(val_valid) * 100
        print(f"  Conf {conf:.0%}: {len(val_valid)} bets, ROI: {val_roi:+.1f}%")

print("[6/7] Testing on 2025...")
X_test = test_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X_test_scaled = scaler.transform(X_test)

test_df['PredProb'] = model.predict_proba(X_test_scaled)[:, 1]
test_df['MarginOverSecond'] = test_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
test_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]

print("\n" + "="*70)
print("V15 RESULTS - 2025 OUT-OF-SAMPLE")
print("="*70)

print("\nBy Confidence ($3-$8):")
for conf in [0.05, 0.08, 0.10, 0.12, 0.15]:
    test_strat = test_leaders[
        (test_leaders['MarginOverSecond'] >= conf) & 
        (test_leaders['BSP'] >= 3) & (test_leaders['BSP'] <= 8)
    ]
    test_valid = test_strat.dropna(subset=['BSP'])
    if len(test_valid) > 0:
        wins = test_valid['Won'].sum()
        sr = wins / len(test_valid) * 100
        test_profit = (test_valid[test_valid['Won']==1]['BSP'].sum() * 0.9) - len(test_valid)
        test_roi = test_profit / len(test_valid) * 100
        status = "✓" if test_roi >= 5 else ""
        print(f"  Conf {conf:.0%}: {len(test_valid):>5} bets, SR: {sr:>5.1f}%, ROI: {test_roi:>+6.1f}% {status}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Save model
print("\n[7/7] Saving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'features': FEATURE_COLS,
    'params': v15_params
}
with open('models/pace_v15_pir.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved to models/pace_v15_pir.pkl")
print("="*70)
