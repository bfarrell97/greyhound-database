"""
V16 Ratings Model - Fair Price Prediction
Key Concept: Predict true probability for EACH runner (not just winner)
Bet when BSP > Model's Fair Price (value betting)
Uses 100 iterations for fast testing
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
print("V16 RATINGS MODEL - VALUE BETTING")
print("Train: 2020-2023 | Val: 2024 | Test: 2025")
print("Uses 100 iterations for fast testing")
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

print(f"[1/7] Loaded {len(df):,} entries")

print("[2/7] Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})

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
            positions = [h['position'] for h in recent if h['position'] is not None]
            splits = [h['split'] for h in recent if h['split'] is not None]
            
            if len(times) >= 3:
                features = {
                    'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'],
                    'Distance': distance, 'MeetingDate': race_date
                }
                
                # Core Features (kept focused for speed)
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
                    val_rows.append(features)
                else:
                    test_rows.append(features)
    
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
val_df = pd.DataFrame(val_rows)
test_df = pd.DataFrame(test_rows)

exclude_cols = ['RaceID', 'Won', 'BSP', 'Distance', 'MeetingDate']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

print("[3/7] Training model with probability calibration...")
X_train = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['Won']
X_val = val_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_val = val_df['Won']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Fast training with 100 iterations
base_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=40,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
base_model.fit(X_train_scaled, y_train)

# Calibrate probabilities using isotonic regression (cv=3 for cross-validation)
print("  Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train_scaled, y_train)

print("[4/7] Generating fair prices on validation set...")
val_df['PredProb'] = calibrated_model.predict_proba(X_val_scaled)[:, 1]
val_df['FairPrice'] = 1 / val_df['PredProb'].clip(0.01, 0.99)  # Convert prob to odds

# Value Betting: Bet when BSP > Fair Price
val_df['HasValue'] = val_df['BSP'] > val_df['FairPrice']
val_df['ValueEdge'] = (val_df['BSP'] - val_df['FairPrice']) / val_df['FairPrice']

print("[5/7] Testing value betting on 2024...")
print("\nValue Betting Thresholds (2024):")
for edge in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    value_bets = val_df[(val_df['ValueEdge'] >= edge) & val_df['BSP'].between(2, 20)]
    if len(value_bets) > 50:
        wins = value_bets['Won'].sum()
        sr = wins / len(value_bets) * 100
        profit = (value_bets[value_bets['Won']==1]['BSP'].sum() * 0.9) - len(value_bets)
        roi = profit / len(value_bets) * 100
        print(f"  Edge >= {edge:.0%}: {len(value_bets):>5} bets, SR: {sr:>5.1f}%, ROI: {roi:>+6.1f}%")

print("[6/7] Testing on 2025 (out-of-sample)...")
X_test = test_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X_test_scaled = scaler.transform(X_test)

test_df['PredProb'] = calibrated_model.predict_proba(X_test_scaled)[:, 1]
test_df['FairPrice'] = 1 / test_df['PredProb'].clip(0.01, 0.99)
test_df['HasValue'] = test_df['BSP'] > test_df['FairPrice']
test_df['ValueEdge'] = (test_df['BSP'] - test_df['FairPrice']) / test_df['FairPrice']

print("\n" + "="*70)
print("V16 RESULTS - 2025 OUT-OF-SAMPLE")
print("="*70)

print("\nValue Betting Thresholds ($2-$20 price range):")
for edge in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    value_bets = test_df[(test_df['ValueEdge'] >= edge) & test_df['BSP'].between(2, 20)]
    if len(value_bets) > 50:
        wins = value_bets['Won'].sum()
        sr = wins / len(value_bets) * 100
        profit = (value_bets[value_bets['Won']==1]['BSP'].sum() * 0.9) - len(value_bets)
        roi = profit / len(value_bets) * 100
        status = "✓" if roi >= 5 else ""
        print(f"  Edge >= {edge:.0%}: {len(value_bets):>6} bets, SR: {sr:>5.1f}%, ROI: {roi:>+6.1f}% {status}")

print("\nBy Price Range (10%+ edge):")
for low, high in [(2, 5), (3, 8), (5, 15), (10, 30)]:
    value_bets = test_df[(test_df['ValueEdge'] >= 0.10) & test_df['BSP'].between(low, high)]
    if len(value_bets) > 30:
        wins = value_bets['Won'].sum()
        sr = wins / len(value_bets) * 100
        profit = (value_bets[value_bets['Won']==1]['BSP'].sum() * 0.9) - len(value_bets)
        roi = profit / len(value_bets) * 100
        status = "✓" if roi >= 5 else ""
        print(f"  ${low}-${high}: {len(value_bets):>5} bets, SR: {sr:>5.1f}%, ROI: {roi:>+6.1f}% {status}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Save
print("\n[7/7] Saving model...")
model_data = {
    'model': calibrated_model,
    'base_model': base_model,
    'scaler': scaler,
    'features': FEATURE_COLS
}
with open('models/pace_v16_ratings.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved to models/pace_v16_ratings.pkl")
print("="*70)
