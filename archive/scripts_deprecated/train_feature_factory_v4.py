import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import pearsonr
import itertools

# Configuration
DB_PATH = 'greyhound_racing.db'

print("LOADING Data for V4 Feature Factory (Speed Figures)...")
conn = sqlite3.connect(DB_PATH)
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2024-12-31'
  AND ge.FinishTime IS NOT NULL AND ge.FinishTime > 0
  AND ge.BSP IS NOT NULL
ORDER BY rm.MeetingDate
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Preprocessing
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['Speed'] = df['Distance'] / df['FinishTime']

# --- BEYER SPEED FIGURE LOGIC (Z-Score) ---
print("Calculating Track Standards (Speed Ratings)...")
# Filter outliers (some FinishTimes are 999 or 0.1)
df = df[(df['Speed'] > 5) & (df['Speed'] < 25)] # Normal Greyhound speed ~15-18 m/s

# Group by Track & Distance
stats = df.groupby(['TrackID', 'Distance'])['Speed'].agg(['mean', 'std']).reset_index()
stats.columns = ['TrackID', 'Distance', 'TrackMean', 'TrackStd']

# Merge back
df = df.merge(stats, on=['TrackID', 'Distance'], how='left')

# Calculate Rating (Z-Score * 20 + 100) -> Beyer Scale style
df['SpeedRating'] = ((df['Speed'] - df['TrackMean']) / df['TrackStd']) * 15 + 100
df['SpeedRating'] = df['SpeedRating'].fillna(80) # Default for single-race tracks

print(f"  Speed Rating Mean: {df['SpeedRating'].mean():.2f}")

# --- FEATURE GENERATION ---
print("Mining Features...")
features = []
df = df.sort_values(by=['GreyhoundID', 'MeetingDate'])

# Rolling Calculations (Vectorized for Factory Analysis)
# We will do a loop for correlations, but first generate candidates.
# To generate interaction features, we need the rolling stats in cols.

g = df.groupby('GreyhoundID')
s_rating = g['SpeedRating'].shift(1) # T-1
s_bsp = g['BSP'].shift(1)
s_pos = g['Position'].shift(1)

# 1. Speed Features
for w in [3, 5, 10]:
    df[f'Speed_Mean_{w}'] = s_rating.rolling(w, min_periods=max(1, w-2)).mean()
    df[f'Speed_Max_{w}'] = s_rating.rolling(w, min_periods=max(1, w-2)).max()
    df[f'Speed_Trend_{w}'] = s_rating - df[f'Speed_Mean_{w}'].shift(1) # Last vs Avg

# 2. BSP Features (Re-verify)
for w in [10]:
    df[f'BSP_Mean_{w}'] = s_bsp.rolling(w, min_periods=w).mean()

# 3. Combinations
df['Speed_x_Class'] = df['Speed_Mean_3'] * df['BSP_Mean_10'] # Is fast dog in good class?
df['Speed_div_Class'] = df['Speed_Mean_3'] / (df['BSP_Mean_10'] + 1)
df['Value_Speed'] = df['Speed_Mean_3'] * df['BSP'] # Current Odds * Speed? (Label Leakage? No, examining correlation with LogBSP)
# Wait, we want correlation with LogBSP.
# `Value_Speed` using CURRENT BSP helps identify Overpriced dogs?
# But we Predict LogBSP. We can't use Current BSP as feature.
# So `Value_Speed` is invalid feature.

# Valid Combinations:
df['Trainer_AvgSpeed'] = df.groupby('TrainerID')['SpeedRating'].transform(lambda x: x.shift(1).expanding().mean())

# Define Analysis List
CANDIDATES = [
    'SpeedRating', # Last Race Rating (Shift(1) actually, handled by s_rating logic? No, s_rating is T-1)
                   # Wait, valid feature is `s_rating` (Last Run).
    'Speed_Mean_3', 'Speed_Mean_5', 'Speed_Mean_10',
    'Speed_Max_3', 'Speed_Trend_3',
    'Speed_x_Class', 'Speed_div_Class',
    'Trainer_AvgSpeed'
]

# Correct 'SpeedRating' feature to be T-1
df['Last_SpeedRating'] = s_rating

analysis_df = df.dropna(subset=CANDIDATES + ['LogBSP']).copy()

print(f"Analyzing {len(analysis_df):,} rows...")
print("-" * 50)
print(f"{'Feature':<30} | {'Corr (LogBSP)':<15}")
print("-" * 50)

for f in CANDIDATES + ['Last_SpeedRating']:
    if f not in df.columns: continue
    try:
        corr, _ = pearsonr(analysis_df[f], analysis_df['LogBSP'])
        print(f"{f:<30} | {corr:.4f}")
    except:
        print(f"{f:<30} | Error")

print("-" * 50)
