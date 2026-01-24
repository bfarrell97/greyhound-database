import pandas as pd
import numpy as np
import sqlite3
from autogluon.tabular import TabularPredictor
import os

# Configuration
DB_PATH = 'greyhound_racing.db'
MODEL_PATH = 'models/autogluon_bsp_v24_speed'
TRAIN_ROWS = 2000000 

print("LOADING DATA for V24 (Speed Figures + Class Interaction)...")
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
df['Speed'] = df['Distance'] / df['FinishTime']

# --- SPEED FIGURE CALCULATION (In-Memory) ---
print("Calculating Speed Ratings...")
# Filter valid speeds for stats
valid_speeds = df[(df['Speed'] > 5) & (df['Speed'] < 25)].copy()
stats = valid_speeds.groupby(['TrackID', 'Distance'])['Speed'].agg(['mean', 'std']).reset_index()
stats.columns = ['TrackID', 'Distance', 'TrackMean', 'TrackStd']

# Merge back
df = df.merge(stats, on=['TrackID', 'Distance'], how='left')

# Calculate Rating (Z-Score * 15 + 100)
df['SpeedRating'] = ((df['Speed'] - df['TrackMean']) / df['TrackStd']) * 15 + 100
df['SpeedRating'] = df['SpeedRating'].fillna(80) 

# --- FEATURE ENGINEERING ---
print("Generating Features...")
df = df.sort_values(by=['GreyhoundID', 'MeetingDate'])

g = df.groupby('GreyhoundID')
s_bsp = g['BSP'].shift(1)
s_pos = g['Position'].shift(1)
s_rating = g['SpeedRating'].shift(1) # Last Race Rating

# 1. Base Features
for w in [3, 10]:
    # BSP
    df[f'BSP_Mean_{w}'] = s_bsp.rolling(window=w, min_periods=w).mean()
    df[f'BSP_Std_{w}'] = s_bsp.rolling(window=w, min_periods=w).std()
    
    # Position
    df[f'Pos_Mean_{w}'] = s_pos.rolling(window=w, min_periods=w).mean()
    
    # Speed
    df[f'Speed_Mean_{w}'] = s_rating.rolling(window=w, min_periods=max(1, w-2)).mean()
    df[f'Speed_Max_{w}'] = s_rating.rolling(window=w, min_periods=max(1, w-2)).max()

# 2. Context Stats
df['Trainer_AvgBSP'] = df.groupby('TrainerID')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)
df['BoxTrack'] = df['TrackID'].astype(str) + '_' + df['Box'].astype(str)
df['Box_Track_AvgBSP'] = df.groupby('BoxTrack')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)

# 3. Interactions (The Secret Sauce)
df['LastBSP'] = s_bsp
df['Last_SpeedRating'] = s_rating

# Speed / Class Ratio (If dog is fast but class (MeanBSP) is high/weak, or vice versa)
# MeanBSP: Higher = Weaker Class.
# Speed: Higher = Faster.
# Ratio: Speed / MeanBSP. 
# Strong Dog (High Speed) in Weak Field (High BSP)?
# Actually, Dog's OWN MeanBSP reflects its class level. 
# If Dog has MeanBSP 20 (Longshot) but Speed 110 (Fast), it's undervalued?
df['Speed_div_Class'] = df['Speed_Mean_3'] / (df['BSP_Mean_10'] + 1)
df['Speed_x_Class'] = df['Speed_Mean_3'] * df['BSP_Mean_10']

# Features List
FEATURES = [
    'BSP_Mean_3', 'BSP_Mean_10', 'BSP_Std_10',
    'Pos_Mean_3', 'Pos_Mean_10',
    'Speed_Mean_3', 'Speed_Mean_10', 'Speed_Max_3',
    'Last_SpeedRating',
    'Speed_div_Class', 'Speed_x_Class',
    'LastBSP',
    'Trainer_AvgBSP', 'Box_Track_AvgBSP',
    'Box', 'Distance'
]

# Clean NaNs
df[FEATURES] = df[FEATURES].fillna(0)

# Date Split
train_data = df[df['MeetingDate'] < '2024-01-01']
test_data = df[df['MeetingDate'] >= '2024-01-01']

print(f"Training on {len(train_data):,} rows. Testing on {len(test_data):,} rows.")

# Train
print(f"Training AutoGluon V24 (Speed + Best Quality) on {len(FEATURES)} features...")
predictor = TabularPredictor(label='LogBSP', path=MODEL_PATH, eval_metric='mean_absolute_error').fit(
    train_data[FEATURES + ['LogBSP']],
    presets='best_quality',   
    time_limit=1200 # 20 mins
)

# Evaluate
print("Evaluation (Test Set):")
perf = predictor.evaluate(test_data[FEATURES + ['LogBSP']])

preds_log = predictor.predict(test_data[FEATURES])
preds_bsp = np.exp(preds_log)
actual_bsp = np.exp(test_data['LogBSP'])
mape = np.mean(np.abs((actual_bsp - preds_bsp) / actual_bsp)) * 100
print(f"MAPE (BSP): {mape:.2f}%")

try:
    fi = predictor.feature_importance(test_data[FEATURES + ['LogBSP']].sample(5000))
    print("\nFeature Importance:")
    print(fi.head(10))
except:
    pass
