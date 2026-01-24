"""
AutoGluon AutoML Model for Greyhound Racing
Uses automated model selection, hyperparameter tuning, and ensembling
Train: 2020-2023 | Val: 2024 | Test: 2025
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
import time
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

print("="*70)
print("AUTOGLUON AUTOML MODEL")
print("Train: 2020-2023 | Test on 2024 + 2025")
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

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
            'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition', 'Margin']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']

print(f"[1/5] Loaded {len(df):,} entries")

print("[2/5] Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})

train_rows = []
test_24_rows = []
test_25_rows = []
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
            positions = [h['position'] for h in recent if h['position'] is not None]
            splits = [h['split'] for h in recent if h['split'] is not None]
            
            if len(times) >= 3:
                features = {
                    'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'],
                    'Distance': distance, 'MeetingDate': race_date
                }
                
                # Core Features
                features['TimeBest'] = min(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeStd'] = np.std(times)
                features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                
                features['PosAvg'] = np.mean(positions)
                features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                features['PlaceRate5'] = sum(1 for p in positions[-5:] if p <= 3) / min(5, len(positions))
                features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                features['CareerStarts'] = min(len(hist), 100)
                features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                
                box = int(r['Box']) if pd.notna(r['Box']) else 4
                features['Box'] = box
                
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                
                if year <= 2023:
                    train_rows.append(features)
                elif year == 2024:
                    test_24_rows.append(features)
                else:
                    test_25_rows.append(features)
    
    # Update history
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        box = int(r['Box']) if pd.notna(r['Box']) else 4
        won = r['Won']
        
        dog_history[dog_id].append({
            'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'position': r['Position'], 'track_id': track_id, 'distance': distance,
            'split': r['Split'] if pd.notna(r['Split']) else None
        })
        
        if pd.notna(r['TrainerID']):
            tid = r['TrainerID']
            trainer_all[tid]['runs'] += 1
            if won: trainer_all[tid]['wins'] += 1
    
    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

train_df = pd.DataFrame(train_rows)
test_24_df = pd.DataFrame(test_24_rows)
test_25_df = pd.DataFrame(test_25_rows)

exclude_cols = ['RaceID', 'Won', 'BSP', 'Distance', 'MeetingDate']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} | Test 2024: {len(test_24_df):,} | Test 2025: {len(test_25_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

# Prepare data for AutoGluon
train_ag = train_df[FEATURE_COLS + ['Won']].copy()
test_24_ag = test_24_df[FEATURE_COLS].copy()
test_25_ag = test_25_df[FEATURE_COLS].copy()

print("[3/5] Training AutoGluon model (5 min time limit)...")
predictor = TabularPredictor(
    label='Won',
    eval_metric='roc_auc',
    path='models/autogluon_v1'
).fit(
    train_data=train_ag,
    time_limit=300,  # 5 minutes
    presets='best_quality',
    verbosity=2
)

print("\n[4/5] Testing on 2024 and 2025...")

# 2024 predictions
test_24_df['PredProb'] = predictor.predict_proba(test_24_ag)[1]
test_24_df['MarginOverSecond'] = test_24_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
leaders_24 = test_24_df.loc[test_24_df.groupby('RaceID')['PredProb'].idxmax()]

# 2025 predictions
test_25_df['PredProb'] = predictor.predict_proba(test_25_ag)[1]
test_25_df['MarginOverSecond'] = test_25_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
leaders_25 = test_25_df.loc[test_25_df.groupby('RaceID')['PredProb'].idxmax()]

print("\n" + "="*70)
print("AUTOGLUON RESULTS")
print("="*70)

def test_strategy(leaders, name, conf, price_low, price_high):
    strat = leaders[
        (leaders['MarginOverSecond'] >= conf) & 
        (leaders['BSP'] >= price_low) & (leaders['BSP'] <= price_high)
    ]
    valid = strat.dropna(subset=['BSP'])
    if len(valid) < 30:
        return None
    wins = valid['Won'].sum()
    sr = wins / len(valid) * 100
    profit = (valid[valid['Won']==1]['BSP'].sum() * 0.9) - len(valid)
    roi = profit / len(valid) * 100
    return {'name': name, 'bets': len(valid), 'sr': sr, 'roi': roi}

print("\n2024 Results:")
for conf in [0.05, 0.08, 0.10, 0.12]:
    r = test_strategy(leaders_24, f"Conf {conf:.0%} $3-$8", conf, 3, 8)
    if r:
        print(f"  {r['name']}: {r['bets']} bets, SR: {r['sr']:.1f}%, ROI: {r['roi']:+.1f}%")

print("\n2025 Results (OUT-OF-SAMPLE):")
for conf in [0.05, 0.08, 0.10, 0.12]:
    r = test_strategy(leaders_25, f"Conf {conf:.0%} $3-$8", conf, 3, 8)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  {r['name']}: {r['bets']} bets, SR: {r['sr']:.1f}%, ROI: {r['roi']:+.1f}% {status}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Show feature importance
print("\n[5/5] Feature Importance:")
importance = predictor.feature_importance(train_ag)
print(importance.head(10))

print("="*70)
print("Model saved to models/autogluon_v1/")
print("="*70)
