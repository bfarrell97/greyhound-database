"""Test SP bets that also have Price5Min - compare all 3 prices"""
import sqlite3
import pandas as pd
import pickle

conn = sqlite3.connect('greyhound_racing.db')
df = pd.read_sql_query("""
SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
       ge.FinishTime, ge.Position, ge.StartingPrice, ge.Price5Min, ge.BSP
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2020-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.StartingPrice IS NOT NULL
""", conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['SP'] = df['StartingPrice'].astype(str).str.replace('$', '').str.replace(',', '')
df['SP'] = pd.to_numeric(df['SP'], errors='coerce')
df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
df = df[df['SP'].notna() & (df['SP'] > 1)]

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
df['FieldSize'] = df.groupby('RaceKey')['SP'].transform('count')
df = df[df['FieldSize'] >= 6]
df = df.sort_values(['RaceKey', 'PredPace'])
df['Rank'] = df.groupby('RaceKey').cumcount() + 1
df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
df['Gap'] = df['NextTime'] - df['PredPace']

leaders = df[df['Rank'] == 1]

# SP strategy pool: Gap >= 0.15, Middle Dist, SP 3-8
sp_pool = leaders[
    (leaders['Gap'] >= 0.15) &
    (leaders['Distance'] >= 400) & (leaders['Distance'] < 550) &
    (leaders['SP'] >= 3) & (leaders['SP'] <= 8)
].copy()

print(f'SP Strategy Pool: {len(sp_pool)} bets')

# Filter to those with Price5Min
p5_pool = sp_pool[sp_pool['Price5Min'].notna()].copy()
print(f'With Price5Min: {len(p5_pool)} bets')

# Calculate profits at each price
p5_pool['Profit_SP'] = p5_pool.apply(lambda x: (x['SP'] - 1) if x['Position'] == '1' else -1, axis=1)
p5_pool['Profit_P5'] = p5_pool.apply(lambda x: (x['Price5Min'] - 1) if x['Position'] == '1' else -1, axis=1)
p5_pool['Profit_BSP'] = p5_pool.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' and pd.notna(x['BSP']) else -1, axis=1)

print()
print('='*70)
print('COMPARISON: Same bets at SP, Price5Min, and BSP')
print('='*70)

total = len(p5_pool)
wins = p5_pool[p5_pool['Position'] == '1'].shape[0]
strike = wins/total*100 if total > 0 else 0

print(f'\nTotal Bets: {total}')
print(f'Wins: {wins} ({strike:.1f}%)')

print(f'\nAvg SP: ${p5_pool["SP"].mean():.2f}')
print(f'Avg Price5Min: ${p5_pool["Price5Min"].mean():.2f}')
print(f'Avg BSP: ${p5_pool["BSP"].mean():.2f}')

sp_profit = p5_pool['Profit_SP'].sum()
p5_profit = p5_pool['Profit_P5'].sum()
bsp_profit = p5_pool['Profit_BSP'].sum()

print()
print(f'{"Price":<15} {"Profit":<12} {"ROI%":<10}')
print('-'*40)
print(f'{"SP":<15} ${sp_profit:<10.1f} {sp_profit/total*100:<+8.1f}%')
print(f'{"Price5Min":<15} ${p5_profit:<10.1f} {p5_profit/total*100:<+8.1f}%')
print(f'{"BSP":<15} ${bsp_profit:<10.1f} {bsp_profit/total*100:<+8.1f}%')
