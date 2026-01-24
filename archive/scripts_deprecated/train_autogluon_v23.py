import pandas as pd
import numpy as np
import sqlite3
from autogluon.tabular import TabularPredictor
import os

# Configuration
DB_PATH = 'greyhound_racing.db'
MODEL_PATH = 'models/autogluon_bsp_v23_best'
TRAIN_ROWS = 2000000 

print("LOADING DATA for V23 (Vectorized + Best Quality)...")
conn = sqlite3.connect(DB_PATH)
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID, g.DamID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
  AND ge.Position IS NOT NULL 
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Preprocessing
if len(df) > TRAIN_ROWS:
    df = df.tail(TRAIN_ROWS).copy()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')

# --- FEATURE ENGINEERING (Vectorized) ---
print("Generating Features...")
df = df.sort_values(by=['GreyhoundID', 'MeetingDate'])

g = df.groupby('GreyhoundID')
s_bsp = g['BSP'].shift(1) # T-1
s_pos = g['Position'].shift(1)

# Feature Lists
for w in [3, 10]:
    # BSP Stats
    df[f'BSP_Mean_{w}'] = s_bsp.rolling(window=w, min_periods=w).mean()
    df[f'BSP_Min_{w}'] = s_bsp.rolling(window=w, min_periods=w).min()
    df[f'BSP_Max_{w}'] = s_bsp.rolling(window=w, min_periods=w).max()
    df[f'BSP_Std_{w}'] = s_bsp.rolling(window=w, min_periods=w).std()
    
    # Position Stats
    df[f'Pos_Mean_{w}'] = s_pos.rolling(window=w, min_periods=w).mean()

# Context Stats (Expanding Mean)
# Trainer
df['Trainer_AvgBSP'] = df.groupby('TrainerID')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)

# Sire
df['Sire_AvgBSP'] = df.groupby('SireID')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)

# Box * Track
df['BoxTrack'] = df['TrackID'].astype(str) + '_' + df['Box'].astype(str)
df['Box_Track_AvgBSP'] = df.groupby('BoxTrack')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)

# Class Trend (Ratio)
df['LastBSP'] = s_bsp
df['Class_Trend_10'] = df['LastBSP'] / df['BSP_Mean_10']

# Features List
FEATURES = [
    'BSP_Mean_3', 'BSP_Min_3', 'BSP_Max_3', 'BSP_Std_3',
    'BSP_Mean_10', 'BSP_Min_10', 'BSP_Max_10', 'BSP_Std_10',
    'Pos_Mean_3', 'Pos_Mean_10',
    'Trainer_AvgBSP', 'Sire_AvgBSP', 'Box_Track_AvgBSP',
    'Class_Trend_10', 'LastBSP', 'Box', 'Distance'
]

# Clean NaNs
df[FEATURES] = df[FEATURES].fillna(0) # Simple fill

# Date Split
train_data = df[df['MeetingDate'] < '2024-01-01']
test_data = df[df['MeetingDate'] >= '2024-01-01']

print(f"Training on {len(train_data):,} rows. Testing on {len(test_data):,} rows.")

# Train
print("Training AutoGluon V23 (Best Quality)...")
predictor = TabularPredictor(label='LogBSP', path=MODEL_PATH, eval_metric='mean_absolute_error').fit(
    train_data[FEATURES + ['LogBSP']],
    presets='best_quality',   # CHANGED to best_quality
    time_limit=900            # Increased to 15 mins
)

# Evaluate
print("Evaluation (Test Set):")
perf = predictor.evaluate(test_data[FEATURES + ['LogBSP']])

# Calculate BSP MAPE
preds = predictor.predict(test_data[FEATURES])
preds_bsp = np.exp(preds)
actual_bsp = np.exp(test_data['LogBSP'])
mape = np.mean(np.abs((actual_bsp - preds_bsp) / actual_bsp)) * 100
print(f"MAPE (BSP): {mape:.2f}%")

# Feature Importance
try:
    fi = predictor.feature_importance(test_data[FEATURES + ['LogBSP']].sample(5000))
    print("\nFeature Importance:")
    print(fi.head(10))
except:
    pass
