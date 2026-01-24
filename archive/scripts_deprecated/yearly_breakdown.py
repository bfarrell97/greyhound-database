"""Yearly Breakdown: Gap >= 0.15, Middle Dist (400-550m), Odds $3-$8, All Boxes"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

print('Loading and preparing data for yearly breakdown...')
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

benchmarks = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
df['NormTime'] = df['FinishTime'] - df['TrackDistMedian']

df = df.sort_values(['GreyhoundID', 'MeetingDate'])
g = df.groupby('GreyhoundID')
df['Lag1'] = g['NormTime'].shift(1)
df['Lag2'] = g['NormTime'].shift(2)
df['Lag3'] = g['NormTime'].shift(3)
df['Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
df['Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
df['PrevDate'] = g['MeetingDate'].shift(1)
df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)

df = df.dropna(subset=['Roll5']).copy()

with open(PACE_MODEL_PATH, 'rb') as f: model = pickle.load(f)
X = df[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']].copy()
df['PredPace'] = model.predict(X) + df['TrackDistMedian']

df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
df['FieldSize'] = df.groupby('RaceKey')['BSP'].transform('count')
df = df[df['FieldSize'] >= 6]

df = df.sort_values(['RaceKey', 'PredPace'])
df['Rank'] = df.groupby('RaceKey').cumcount() + 1
df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
df['Gap'] = df['NextTime'] - df['PredPace']

leaders = df[df['Rank'] == 1].copy()

# Strategy: Gap >= 0.15, Middle Distance (400-550), Odds 3-8, All Boxes
strat = leaders[
    (leaders['Gap'] >= 0.15) &
    (leaders['Distance'] >= 400) & (leaders['Distance'] < 550) &
    (leaders['BSP'] >= 3.0) & (leaders['BSP'] <= 8.0) &
    leaders['BSP'].notna()
].copy()

strat['Profit'] = strat.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
strat['Year'] = strat['MeetingDate'].dt.year

print()
print('='*70)
print('STRATEGY: Gap >= 0.15, Middle Dist (400-550m), Odds $3-$8, All Boxes')
print('='*70)

print()
print('YEARLY BREAKDOWN:')
print('Year  | Bets | Wins | Strike% | Profit | ROI%')
print('-'*50)

for year in sorted(strat['Year'].unique()):
    yr = strat[strat['Year'] == year]
    wins = yr[yr['Position'] == '1'].shape[0]
    profit = yr['Profit'].sum()
    roi = (profit / len(yr)) * 100 if len(yr) > 0 else 0
    strike = (wins / len(yr)) * 100 if len(yr) > 0 else 0
    print(f'{year}  | {len(yr):<4} | {wins:<4} | {strike:<7.1f} | {profit:<6.1f} | {roi:<6.1f}')

total_bets = len(strat)
total_wins = strat[strat['Position'] == '1'].shape[0]
total_profit = strat['Profit'].sum()
total_roi = (total_profit / total_bets) * 100
total_strike = (total_wins / total_bets) * 100
print('-'*50)
print(f'TOTAL | {total_bets:<4} | {total_wins:<4} | {total_strike:<7.1f} | {total_profit:<6.1f} | {total_roi:<6.1f}')
