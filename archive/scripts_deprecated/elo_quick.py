"""Quick Elo Test: Train 2024, Test 2025"""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

print('='*70)
print('QUICK ELO TEST: Train 2024, Test 2025')
print('='*70)

conn = sqlite3.connect('greyhound_racing.db')

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, r.Distance, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate BETWEEN '2024-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
ORDER BY rm.MeetingDate
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = df['Position'] == 1
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')

print(f"Loaded {len(df):,} entries (2024-2025)")

# Elo calculation
K = 32
ratings = defaultdict(lambda: 1500)
predictions = []

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 2:
        continue
    
    race_date = race_df['MeetingDate'].iloc[0]
    race_ratings = {row['GreyhoundID']: ratings[row['GreyhoundID']] for _, row in race_df.iterrows()}
    
    total_exp = sum(np.exp(r / 400) for r in race_ratings.values())
    expected = {d: np.exp(race_ratings[d] / 400) / total_exp for d in race_ratings}
    pred_winner = max(race_ratings, key=race_ratings.get)
    
    sorted_r = sorted(race_ratings.values(), reverse=True)
    gap = sorted_r[0] - sorted_r[1] if len(sorted_r) >= 2 else 0
    
    if race_date >= datetime(2025, 1, 1):
        for _, row in race_df.iterrows():
            if row['GreyhoundID'] == pred_winner:
                predictions.append({
                    'Rating': race_ratings[row['GreyhoundID']],
                    'Gap': gap, 'Won': row['Won'], 'BSP': row['BSP'], 'Distance': row['Distance']
                })
    
    for _, row in race_df.iterrows():
        dog_id = row['GreyhoundID']
        actual = 1.0 if row['Won'] else 0.0
        ratings[dog_id] += K * (actual - expected[dog_id])

print(f"Predictions for 2025: {len(predictions):,}")

pred_df = pd.DataFrame(predictions)
pred_df = pred_df.dropna(subset=['BSP'])

print('\n' + '='*70)
print('RESULTS')
print('='*70)

wins = pred_df['Won'].sum()
print(f"All Elo Leaders: {len(pred_df):,} bets, {wins:,} wins ({wins/len(pred_df)*100:.1f}%)")

f = pred_df[(pred_df['BSP'] >= 3) & (pred_df['BSP'] <= 8)]
if len(f) > 0:
    wins = f['Won'].sum()
    ret = f[f['Won']]['BSP'].sum()
    profit = ret - len(f)
    roi = profit / len(f) * 100
    print(f"$3-$8: {len(f):,} bets, {wins:,} wins ({wins/len(f)*100:.1f}%), Profit: {profit:.1f}u, ROI: {roi:.1f}%")

g = f[f['Gap'] >= 50]
if len(g) > 0:
    wins = g['Won'].sum()
    ret = g[g['Won']]['BSP'].sum()
    profit = ret - len(g)
    roi = profit / len(g) * 100
    print(f"$3-$8 + Gap>=50: {len(g):,} bets, {wins:,} wins ({wins/len(g)*100:.1f}%), Profit: {profit:.1f}u, ROI: {roi:.1f}%")

m = g[(g['Distance'] >= 400) & (g['Distance'] < 550)]
if len(m) > 0:
    wins = m['Won'].sum()
    ret = m[m['Won']]['BSP'].sum()
    profit = ret - len(m)
    roi = profit / len(m) * 100
    print(f"$3-$8 + Gap>=50 + Mid(400-550): {len(m):,} bets, {wins:,} wins ({wins/len(m)*100:.1f}%), Profit: {profit:.1f}u, ROI: {roi:.1f}%")

print('\n' + '='*70)
