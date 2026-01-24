"""Metro Elo+Pace BOTH AGREE - 2024+2025 Test at $3-$10 (SP)"""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
K = 40

conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query('SELECT TrackName, Distance, MedianTime FROM Benchmarks', conn)
track_list = "', '".join(METRO_TRACKS)

# Load 2023-2025 data (train on 2023, test on 2024-2025)
query = f"""SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.StartingPrice as SP, r.Distance, rm.MeetingDate, t.TrackName
FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate BETWEEN '2023-01-01' AND '2025-11-30' AND ge.Position NOT IN ('SCR', 'DNF', '') AND t.TrackName IN ('{track_list}') ORDER BY rm.MeetingDate"""
df = pd.read_sql_query(query, conn)
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = df['Position'] == 1
df['SP'] = pd.to_numeric(df['SP'], errors='coerce')
df['Year'] = df['MeetingDate'].dt.year

print(f"Loaded {len(df):,} Metro entries (2023-2025)")

# Historical pace data for Roll5
hist_query = """SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate < '2024-01-01' AND ge.FinishTime IS NOT NULL ORDER BY ge.GreyhoundID, rm.MeetingDate"""
hist_df = pd.read_sql_query(hist_query, conn)
conn.close()

hist_df['MeetingDate'] = pd.to_datetime(hist_df['MeetingDate'])
hist_df = hist_df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['MedianTime']
hist_df = hist_df.dropna(subset=['NormTime']).sort_values(['GreyhoundID', 'MeetingDate'])
g = hist_df.groupby('GreyhoundID')
hist_df['Roll5'] = g['NormTime'].transform(lambda x: x.rolling(5, min_periods=5).mean())
hist_df = hist_df.dropna(subset=['Roll5'])
lookup = {row['GreyhoundID']: row['Roll5'] for _, row in hist_df.groupby('GreyhoundID').last().reset_index().iterrows()}

print(f"Dogs with pace history: {len(lookup):,}")

elo = defaultdict(lambda: 1500)
preds = []

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 4:
        continue
    race_date = race_df['MeetingDate'].iloc[0]
    year = race_date.year
    
    race_elo = {r['GreyhoundID']: elo[r['GreyhoundID']] for _, r in race_df.iterrows()}
    pace = {r['GreyhoundID']: lookup[r['GreyhoundID']] for _, r in race_df.iterrows() if r['GreyhoundID'] in lookup}
    
    if len(pace) < 4:
        continue
    
    elo_ldr = max(race_elo, key=race_elo.get)
    pace_ldr = min(pace, key=pace.get)
    
    # Test on 2024 and 2025
    if race_date >= datetime(2024, 1, 1) and elo_ldr == pace_ldr:
        for _, r in race_df.iterrows():
            if r['GreyhoundID'] == elo_ldr:
                preds.append({'Won': r['Won'], 'SP': r['SP'], 'Year': year})
    
    # Update Elo
    total = sum(np.exp(v/400) for v in race_elo.values())
    for _, r in race_df.iterrows():
        actual = 1 if r['Won'] else 0
        expected = np.exp(race_elo[r['GreyhoundID']]/400) / total
        elo[r['GreyhoundID']] += K * (actual - expected)

pred_df = pd.DataFrame(preds).dropna(subset=['SP'])

# Filter to $3-$10
f = pred_df[(pred_df['SP'] >= 3) & (pred_df['SP'] < 10)]

print('='*60)
print('METRO ELO+PACE BOTH AGREE @ $3-$10 (SP)')
print('='*60)
print(f'Total: {len(f)} bets, {f["Won"].sum()} wins ({f["Won"].mean()*100:.1f}%)')
ret = f[f['Won']]['SP'].sum()
profit = ret - len(f)
roi = profit / len(f) * 100
print(f'Profit: {profit:.1f}u, ROI: {roi:+.1f}%')
print()

# By year
print('BY YEAR:')
print(f"{'Year':<8} {'Bets':>6} {'Wins':>6} {'SR%':>7} {'Profit':>8} {'ROI%':>7}")
print('-'*50)
for year in sorted(f['Year'].unique()):
    y = f[f['Year'] == year]
    w = y['Won'].sum()
    sr = w/len(y)*100
    ret = y[y['Won']]['SP'].sum()
    p = ret - len(y)
    roi = p/len(y)*100
    print(f'{year:<8} {len(y):>6} {w:>6} {sr:>6.1f}% {p:>7.1f}u {roi:>6.1f}%')

print('='*60)
