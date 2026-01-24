"""Test User's Specific Config: Gap >= 0.10, Odds $2-$10, Dist < 550, Inside/Outside Box"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

print("Loading and preparing data...")
conn = sqlite3.connect(DB_PATH)
query = """
SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
       ge.FinishTime, ge.Position, ge.BSP, COALESCE(ge.PrizeMoney, 0) as PrizeMoney
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2020-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')

# Benchmarks
benchmarks = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
df['NormTime'] = df['FinishTime'] - df['TrackDistMedian']

df = df.sort_values(['GreyhoundID', 'MeetingDate'])
g = df.groupby('GreyhoundID')

# V1 Features
df['Lag1'] = g['NormTime'].shift(1)
df['Lag2'] = g['NormTime'].shift(2)
df['Lag3'] = g['NormTime'].shift(3)
df['Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
df['Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
df['PrevDate'] = g['MeetingDate'].shift(1)
df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)

df = df.dropna(subset=['Roll5']).copy()

# Predict
with open(PACE_MODEL_PATH, 'rb') as f: model = pickle.load(f)
X = df[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']].copy()
df['PredPace'] = model.predict(X) + df['TrackDistMedian']

# Ranking
df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
df['FieldSize'] = df.groupby('RaceKey')['BSP'].transform('count')
df = df[df['FieldSize'] >= 6]

df = df.sort_values(['RaceKey', 'PredPace'])
df['Rank'] = df.groupby('RaceKey').cumcount() + 1
df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
df['Gap'] = df['NextTime'] - df['PredPace']

# Filter leaders
leaders = df[df['Rank'] == 1].copy()
leaders = leaders[leaders['BSP'].notna() & (leaders['BSP'] > 1)]

# Date range
date_range = (leaders['MeetingDate'].max() - leaders['MeetingDate'].min()).days

print(f"Total Leaders: {len(leaders)}, Date Range: {date_range} days")

# User's config: Dist < 550, Inside (1-3) OR Outside (6-8)
base = leaders[
    (leaders['Distance'] < 550) &
    ((leaders['Box'].isin([1, 2, 3])) | (leaders['Box'].isin([6, 7, 8])))
]

print(f"After Dist < 550 + Inside/Outside Box: {len(base)} leaders")

# Test multiple combinations
print("\n" + "="*85)
print("USER CONFIG: Gap >= X, Distance < 550m, Box Inside(1-3) OR Outside(6-8), BSP $X-$Y")
print("="*85)
print(f"{'Gap':<5} | {'MinOdds':<7} | {'MaxOdds':<7} | {'Bets':<6} | {'BetsPerDay':<10} | {'Strike%':<8} | {'Profit':<8} | {'ROI%':<8}")
print("-"*85)

for gap in [0.10, 0.12, 0.15, 0.18]:
    for min_o in [2.0, 2.2, 2.5, 3.0]:
        for max_o in [5, 6, 8, 10]:
            filt = base[
                (base['Gap'] >= gap) &
                (base['BSP'] >= min_o) &
                (base['BSP'] <= max_o)
            ].copy()
            
            if len(filt) < 300:
                continue
                
            filt['Profit'] = filt.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
            
            wins = filt[filt['Position'] == '1'].shape[0]
            bpd = len(filt) / date_range
            roi = (filt['Profit'].sum() / len(filt)) * 100 if len(filt) > 0 else 0
            
            if roi >= 5:  # Only show profitable
                print(f"{gap:<5} | ${min_o:<6} | ${max_o:<6} | {len(filt):<6} | {bpd:<10.2f} | {(wins/len(filt))*100:<8.1f} | {filt['Profit'].sum():<8.1f} | {roi:<8.1f}")

print("\nDone!")
