"""
AutoGluon V19 - BSP Prediction Accuracy
Focus: Accurately predict BSP using V12 features + Historical BSP data
Metric: MAE (Mean Absolute Error) and MAPE both on Log and Price scale
"""
import sqlite3
import pandas as pd
import numpy as np
import time
import pickle
import random
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("AUTOGLUON V19 - BSP PREDICTION")
print("Target: Predict BSP with high accuracy (minimize MAE/MAPE)")
print("rain: 2020-2023 | Test: 2024 + 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName,
       g.SireID, g.DamID, g.DateOfBirth
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL 
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Year'] = df['MeetingDate'].dt.year
# Log transform BSP for better regression
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))

# Basic cleaning
for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"[1/5] Loaded {len(df):,} entries with BSP")

print("[2/5] Building features (V12 + BSP History)...")
# Simply use V12 feature logic but add BSP history
from collections import defaultdict
dog_history = defaultdict(list)

# We'll re-use the V12 feature loop logic roughly
processed = 0
rows = []

for race_id, race_df in df.groupby('RaceID', sort=False):
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_history.get(dog_id, [])
        
        if len(hist) >= 3:
            recent = hist[-10:]
            # Only use if we have BSP history
            bsps = [h['bsp'] for h in recent if h['bsp'] is not None and h['bsp'] > 1]
            times = [h['time'] for h in recent if h['time'] is not None]
            
            if len(bsps) >= 2 and len(times) >= 2:
                features = {
                    'RaceID': race_id, 'GreyhoundID': dog_id,
                    'LogBSP': r['LogBSP'], 'BSP': r['BSP'],
                    'Year': r['Year']
                }
                
                # BSP History Features (Crucial for prediction)
                features['BSPAvg'] = np.mean(bsps)
                features['BSPAvg3'] = np.mean(bsps[-3:])
                features['BSPLag1'] = bsps[-1]
                features['BSPLag2'] = bsps[-2] if len(bsps) >= 2 else bsps[-1]
                features['BSPMin'] = min(bsps)
                features['BSPMax'] = max(bsps)
                features['BSPStd'] = np.std(bsps) if len(bsps) >= 3 else 0
                features['BSPTrend'] = bsps[-1] - bsps[0] if len(bsps) >= 2 else 0
                
                # Log BSP features
                log_bsps = [np.log(b) for b in bsps]
                features['LogBSPAvg'] = np.mean(log_bsps)
                features['LogBSPLag1'] = log_bsps[-1]
                
                # Performance Features (V12 subset)
                features['TimeBest'] = min(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeLag1'] = times[-1]
                
                # Careful with types for numpy mean
                positions = [float(h['pos']) for h in recent if h['pos'] and str(h['pos']).isdigit()]
                features['PosAvg'] = np.mean(positions) if positions else 4.5
                
                features['CareerStarts'] = len(hist)
                features['DaysSince'] = (race_date - hist[-1]['date']).days
                features['Box'] = int(r['Box']) if pd.notna(r['Box']) else 4
                
                rows.append(features)
        
    # Update history
    for _, r in race_df.iterrows():
        dog_history[r['GreyhoundID']].append({
            'date': race_date, 'pos': r['Position'],
            'time': r['FinishTime'], 'bsp': r['BSP']
        })
    
    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

full_df = pd.DataFrame(rows)
train_df = full_df[full_df['Year'] <= 2023].copy()
test_df = full_df[full_df['Year'] >= 2024].copy()

exclude = ['RaceID', 'GreyhoundID', 'LogBSP', 'BSP', 'Year']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude]

print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

print("[3/5] Training AutoGluon for LogBSP...")
# We use LogBSP as target because prices are log-normal
# (e.g. difference between $2 and $2.20 is same significance as $20 and $22)

model_path = f'models/autogluon_bsp_v19_{random.randint(1000,9999)}'
predictor = TabularPredictor(
    label='LogBSP',
    eval_metric='mean_absolute_error',
    path=model_path,
    problem_type='regression'
).fit(
    train_data=train_df[FEATURE_COLS + ['LogBSP']],
    time_limit=300,  # 5 min training
    presets='best_quality'
)

print("[4/5] Evaluating on Test Data...")
# Predict LogBSP then convert back
preds_log = predictor.predict(test_df[FEATURE_COLS])
test_df['PredLogBSP'] = preds_log
test_df['PredBSP'] = np.exp(preds_log)

# Metrics
mae_log = np.mean(np.abs(test_df['LogBSP'] - test_df['PredLogBSP']))
mae_price = np.mean(np.abs(test_df['BSP'] - test_df['PredBSP']))
mape = np.mean(np.abs(test_df['BSP'] - test_df['PredBSP']) / test_df['BSP']) * 100

print("\n" + "="*70)
print("BSP PREDICTION RESULTS (V19)")
print("="*70)
print(f"Log MAE:   {mae_log:.4f}")
print(f"Price MAE: ${mae_price:.2f}")
print(f"MAPE:      {mape:.1f}%")
print("-" * 30)

# Accuracy by Price Range
for low, high in [(1, 3), (3, 5), (5, 10), (10, 20), (20, 50)]:
    subset = test_df[(test_df['BSP'] >= low) & (test_df['BSP'] < high)]
    if len(subset) > 0:
        sub_mape = np.mean(np.abs(subset['BSP'] - subset['PredBSP']) / subset['BSP']) * 100
        print(f"${low}-${high}: MAPE {sub_mape:.1f}% ({len(subset):,} bets)")

print("-" * 30)
# Top Features
print("Top Features:")
try:
    fi = predictor.feature_importance(test_df[FEATURE_COLS + ['LogBSP']].sample(5000))
    print(fi.head(5))
except:
    pass

print(f"\nModel saved to: {model_path}")
print("="*70)
