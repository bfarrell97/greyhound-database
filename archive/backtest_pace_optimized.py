"""
BACKTEST: Pace-Based Betting Strategy (Optimized)
Provides timestamped progress updates as it validates the +13% ROI claim
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*100)
progress("PACE-BASED BETTING STRATEGY BACKTEST")
progress("="*100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

progress("\nSTEP 1: Loading race data...")
query = """
SELECT ge.GreyhoundID, g.GreyhoundName, t.TrackName, rm.MeetingDate,
       ge.Position, ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate <= '2025-12-09'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
"""

start = time.time()
df = pd.read_sql_query(query, conn)
progress(f"Loaded {len(df):,} entries in {time.time()-start:.1f}s", indent=1)

progress("\nSTEP 2: Filtering to Australian tracks...")
excluded = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
           'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded)]
df = df[~df['TrackName'].str.contains('NZ|TAS', na=False, case=False)]
progress(f"After filter: {len(df):,} entries", indent=1)

progress("\nSTEP 3: Processing results...")
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
progress(f"Winners: {df['IsWinner'].sum():,} (baseline: {df['IsWinner'].mean()*100:.2f}%)", indent=1)

progress("\nSTEP 4: Loading historical benchmarks...")
hist_query = """
SELECT ge.GreyhoundID, rm.MeetingDate,
       ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position NOT IN ('DNF', 'SCR')
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""

start = time.time()
hist = pd.read_sql_query(hist_query, conn)
conn.close()
progress(f"Loaded {len(hist):,} historical entries in {time.time()-start:.1f}s", indent=1)

progress("\nSTEP 5: Building historical pace index...")
hist['MeetingDate'] = pd.to_datetime(hist['MeetingDate'])
hist['Pace'] = hist['FinishTimeBenchmarkLengths'] + hist['MeetingAvgBenchmarkLengths']

# Build dictionary: dog_id -> list of (date, pace) sorted by date
pace_by_dog = {}
for dog_id in hist['GreyhoundID'].unique():
    dog_hist = hist[hist['GreyhoundID'] == dog_id].sort_values('MeetingDate')
    pace_by_dog[dog_id] = list(zip(pd.to_datetime(dog_hist['MeetingDate']), dog_hist['Pace']))

progress(f"Built index for {len(pace_by_dog):,} dogs", indent=1)

progress("\nSTEP 6: Calculating historical pace for each entry...")
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])

def get_hist_pace(dog_id, race_date):
    if dog_id not in pace_by_dog:
        return np.nan
    
    races_before = [p for d, p in pace_by_dog[dog_id] if d < race_date]
    if len(races_before) < 1:
        return np.nan
    
    return np.mean(races_before[-5:])

start = time.time()
paces = []
for idx, (_, row) in enumerate(df.iterrows()):
    if (idx + 1) % 20000 == 0:
        elapsed = time.time() - start
        pct = ((idx + 1) / len(df)) * 100
        eta = (len(df) - idx - 1) * (elapsed / (idx + 1))
        progress(f"{idx+1:,}/{len(df):,} ({pct:.1f}%) ETA {eta:.0f}s", indent=1)
    
    pace = get_hist_pace(row['GreyhoundID'], row['MeetingDate'])
    paces.append(pace)

df['HistoricalPace'] = paces
progress(f"Completed in {time.time()-start:.1f}s", indent=1)

progress("\nSTEP 7: Filtering to optimal odds...")
df_filtered = df[(df['HistoricalPace'].notna()) &
                 (df['StartingPrice'] >= 1.50) &
                 (df['StartingPrice'] <= 2.00)].copy()
baseline = df_filtered['IsWinner'].mean() * 100
progress(f"Optimal odds sample: {len(df_filtered):,} (baseline: {baseline:.2f}%)", indent=1)

progress("\nSTEP 8: Testing pace thresholds...")
results = []
for threshold in [0.0, 0.25, 0.5, 1.0, 1.5]:
    subset = df_filtered[df_filtered['HistoricalPace'] >= threshold]
    if len(subset) == 0:
        continue
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    strike = (wins / bets * 100)
    stake = bets * 1.0
    ret = (subset['StartingPrice'] * subset['IsWinner']).sum()
    roi = ((ret - stake) / stake * 100) if stake > 0 else 0
    profit = ret - stake
    
    results.append({'Threshold': threshold, 'Bets': bets, 'Wins': wins,
                   'Strike%': strike, 'Profit': profit, 'ROI%': roi})
    progress(f"Pace >= {threshold:.2f}: {bets:,} bets, {strike:.2f}% strike, ROI {roi:+.2f}%", indent=1)

progress("\n" + "="*100)
progress("RESULTS SUMMARY")
progress("="*100)

for r in results:
    print(f"  Pace >= {r['Threshold']:.2f} | Bets: {r['Bets']:,} | Strike: {r['Strike%']:.2f}% | ROI: {r['ROI%']:+.2f}%")

progress("\nDETAILED ANALYSIS: Pace >= 0.5")
progress("="*100)

r = [x for x in results if x['Threshold'] == 0.5][0]
progress(f"Bets: {r['Bets']:,} | Wins: {r['Wins']:,} | Strike: {r['Strike%']:.2f}%", indent=1)
progress(f"ROI: {r['ROI%']:+.2f}% vs +13.00% target (Diff: {r['ROI%']-13:+.2f}%)", indent=1)

if r['ROI%'] < 0:
    progress("\n*** CRITICAL: NEGATIVE ROI FOUND ***", indent=0)
    progress("The +13% ROI claim from prior analysis may be incorrect.", indent=0)

progress(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
progress("="*100)
