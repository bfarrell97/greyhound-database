"""
Simple Pace Leader Test using SP (2025 only)
=============================================
Validate pace model performance with Starting Price
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("="*70)
print("PACE LEADER TEST - SP (2025 ONLY)")
print("="*70)

conn = sqlite3.connect('greyhound_racing.db')

# Load pace model
with open('models/pace_xgb_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
pace_model = artifacts['model']

# Load benchmarks
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)

# Load 2025 data only
print("\n[1/3] Loading 2025 race data...")
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.StartingPrice as SP, ge.BSP, ge.Box,
       r.Distance, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate BETWEEN '2025-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
"""
df = pd.read_sql_query(query, conn)

df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = df['Position'] == 1
df['SP'] = pd.to_numeric(df['SP'], errors='coerce')
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)

print(f"Loaded {len(df):,} entries")
print(f"With SP: {df['SP'].notna().sum():,}")
print(f"With BSP: {df['BSP'].notna().sum():,}")

# Load historical pace data for Roll5
print("\n[2/3] Loading historical data for Roll5...")
hist_query = """
SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate < '2025-01-01'
  AND ge.FinishTime IS NOT NULL AND ge.Position NOT IN ('DNF', 'SCR', '')
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""
hist_df = pd.read_sql_query(hist_query, conn)
conn.close()

hist_df['MeetingDate'] = pd.to_datetime(hist_df['MeetingDate'])
hist_df = hist_df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['MedianTime']
hist_df = hist_df.dropna(subset=['NormTime'])

# Get Roll5 for each dog (latest available)
hist_df = hist_df.sort_values(['GreyhoundID', 'MeetingDate'])
g = hist_df.groupby('GreyhoundID')
hist_df['Roll5'] = g['NormTime'].transform(lambda x: x.rolling(5, min_periods=5).mean())
hist_df = hist_df.dropna(subset=['Roll5'])

# Get latest Roll5 per dog
latest = hist_df.groupby('GreyhoundID').last().reset_index()[['GreyhoundID', 'Roll5']]
roll5_lookup = dict(zip(latest['GreyhoundID'], latest['Roll5']))

print(f"Dogs with Roll5 history: {len(roll5_lookup):,}")

# [3/3] Find pace leaders per race
print("\n[3/3] Finding pace leaders...")

predictions = []
races_with_leader = 0

for race_id, race_df in df.groupby('RaceID'):
    if len(race_df) < 4:
        continue
    
    distance = race_df['Distance'].iloc[0]
    
    # Get Roll5 for each dog in race
    race_pace = []
    for _, row in race_df.iterrows():
        dog_id = row['GreyhoundID']
        if dog_id in roll5_lookup:
            race_pace.append({
                'GreyhoundID': dog_id,
                'Roll5': roll5_lookup[dog_id],
                'Won': row['Won'],
                'SP': row['SP'],
                'BSP': row['BSP'],
                'Distance': distance
            })
    
    if len(race_pace) < 4:
        continue
    
    # Sort by Roll5 (lower = faster)
    race_pace = sorted(race_pace, key=lambda x: x['Roll5'])
    
    # Pace leader
    leader = race_pace[0]
    gap = race_pace[1]['Roll5'] - race_pace[0]['Roll5']
    
    leader['PaceGap'] = gap
    leader['IsLeader'] = True
    predictions.append(leader)
    races_with_leader += 1

print(f"Races with pace leader: {races_with_leader:,}")

pred_df = pd.DataFrame(predictions)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Using SP
print("\n--- USING STARTING PRICE (SP) ---")
sp_valid = pred_df[(pred_df['SP'].notna()) & (pred_df['SP'] >= 3) & (pred_df['SP'] <= 8)]
if len(sp_valid) > 0:
    wins = sp_valid['Won'].sum()
    sr = wins / len(sp_valid) * 100
    returns = sp_valid[sp_valid['Won']]['SP'].sum()
    profit = returns - len(sp_valid)
    roi = profit / len(sp_valid) * 100
    print(f"Pace Leader @ $3-$8 SP: {len(sp_valid):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
    
    # With gap filter
    sp_gap = sp_valid[sp_valid['PaceGap'] >= 0.15]
    if len(sp_gap) > 50:
        wins = sp_gap['Won'].sum()
        sr = wins / len(sp_gap) * 100
        returns = sp_gap[sp_gap['Won']]['SP'].sum()
        profit = returns - len(sp_gap)
        roi = profit / len(sp_gap) * 100
        print(f"+ Gap >= 0.15: {len(sp_gap):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
    
    # With mid distance
    sp_mid = sp_gap[(sp_gap['Distance'] >= 400) & (sp_gap['Distance'] < 550)]
    if len(sp_mid) > 50:
        wins = sp_mid['Won'].sum()
        sr = wins / len(sp_mid) * 100
        returns = sp_mid[sp_mid['Won']]['SP'].sum()
        profit = returns - len(sp_mid)
        roi = profit / len(sp_mid) * 100
        print(f"+ Mid Dist (400-550): {len(sp_mid):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")

# Using BSP
print("\n--- USING BSP ---")
bsp_valid = pred_df[(pred_df['BSP'].notna()) & (pred_df['BSP'] >= 3) & (pred_df['BSP'] <= 8)]
if len(bsp_valid) > 0:
    wins = bsp_valid['Won'].sum()
    sr = wins / len(bsp_valid) * 100
    returns = bsp_valid[bsp_valid['Won']]['BSP'].sum()
    profit = returns - len(bsp_valid)
    roi = profit / len(bsp_valid) * 100
    print(f"Pace Leader @ $3-$8 BSP: {len(bsp_valid):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
    
    # With gap filter
    bsp_gap = bsp_valid[bsp_valid['PaceGap'] >= 0.15]
    if len(bsp_gap) > 50:
        wins = bsp_gap['Won'].sum()
        sr = wins / len(bsp_gap) * 100
        returns = bsp_gap[bsp_gap['Won']]['BSP'].sum()
        profit = returns - len(bsp_gap)
        roi = profit / len(bsp_gap) * 100
        print(f"+ Gap >= 0.15: {len(bsp_gap):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")

print("\n" + "="*70)
