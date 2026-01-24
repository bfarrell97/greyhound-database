"""Test: Gap >= 0.15, Middle Dist (400-550m), BSP $3-$8"""
import sqlite3
import pandas as pd
import pickle

conn = sqlite3.connect('greyhound_racing.db')
df = pd.read_sql_query("""
SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
       ge.FinishTime, ge.Position, ge.BSP
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2020-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
""", conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')

bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
bench.columns = ['TrackName', 'Distance', 'TrackDistMedian']
df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
df['NormTime'] = df['FinishTime'] - df['TrackDistMedian']

df = df.sort_values(['GreyhoundID', 'MeetingDate'])
g = df.groupby('GreyhoundID')
df['Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
df['Lag1'] = g['NormTime'].shift(1)
df['Lag2'] = g['NormTime'].shift(2)
df['Lag3'] = g['NormTime'].shift(3)
df['Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
df['PrevDate'] = g['MeetingDate'].shift(1)
df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
df = df.dropna(subset=['Roll5']).copy()

with open('models/pace_xgb_model.pkl', 'rb') as f: model = pickle.load(f)
X = df[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']]
df['PredPace'] = model.predict(X) + df['TrackDistMedian']

df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
df['FieldSize'] = df.groupby('RaceKey')['BSP'].transform('count')
df = df[df['FieldSize'] >= 6]
df = df.sort_values(['RaceKey', 'PredPace'])
df['Rank'] = df.groupby('RaceKey').cumcount() + 1
df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
df['Gap'] = df['NextTime'] - df['PredPace']

leaders = df[df['Rank'] == 1]
leaders = leaders[leaders['BSP'].notna() & (leaders['BSP'] > 1)]

# CONFIG: Gap >= 0.15, Middle Dist (400-550m), BSP 3-8
strat = leaders[
    (leaders['Gap'] >= 0.15) &
    (leaders['Distance'] >= 400) & (leaders['Distance'] < 550) &
    (leaders['BSP'] >= 3) & (leaders['BSP'] <= 8)
].copy()

strat['Profit'] = strat.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
strat['Year'] = strat['MeetingDate'].dt.year

print('='*70)
print('STRATEGY: Gap >= 0.15, Middle Dist (400-550m), BSP $3-$8')
print('='*70)
print('Year  | Bets | Wins | Strike% | Profit | ROI%')
print('-'*50)
for year in sorted(strat['Year'].unique()):
    yr = strat[strat['Year'] == year]
    wins = yr[yr['Position'] == '1'].shape[0]
    profit = yr['Profit'].sum()
    roi = (profit / len(yr)) * 100 if len(yr) > 0 else 0
    strike = (wins / len(yr)) * 100 if len(yr) > 0 else 0
    print(f'{year}  | {len(yr):<4} | {wins:<4} | {strike:<7.1f} | {profit:<6.1f} | {roi:<6.1f}')
print('-'*50)
total = len(strat)
wins = strat[strat['Position'] == '1'].shape[0]
profit = strat['Profit'].sum()
roi = profit/total*100 if total > 0 else 0
strike = wins/total*100 if total > 0 else 0
print(f'TOTAL | {total:<4} | {wins:<4} | {strike:<7.1f} | {profit:<6.1f} | {roi:<6.1f}')
print(f'\nAvg BSP: ${strat["BSP"].mean():.2f}')
