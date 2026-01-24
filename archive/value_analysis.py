"""
Find where the model adds value
Compare model predictions to market odds
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("VALUE ANALYSIS - Where does the model beat the market?")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get pace data
progress("Loading data...", indent=1)
pace_df = pd.read_sql_query("""
WITH dog_pace_history_raw AS (
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalBench,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '2025-11-01'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
)
SELECT 
    GreyhoundID,
    AVG(CASE WHEN RaceNum <= 5 THEN TotalBench END) as RawPaceAvg,
    COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
FROM dog_pace_history_raw
GROUP BY GreyhoundID
HAVING PacesUsed >= 5
""", conn)

form_df = pd.read_sql_query("""
WITH dog_form_raw AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position NOT IN ('DNF', 'SCR')
      AND rm.MeetingDate < '2025-11-01'
)
SELECT 
    GreyhoundID,
    SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
    COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
FROM dog_form_raw
GROUP BY GreyhoundID
""", conn)

races_df = pd.read_sql_query("""
SELECT 
    ge.GreyhoundID,
    r.RaceID,
    ge.StartingPrice,
    DATE(rm.MeetingDate) as RaceDate,
    (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2025-11-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
""", conn)

conn.close()

# Merge
df = races_df.merge(pace_df, on='GreyhoundID', how='left')
df = df.merge(form_df, on='GreyhoundID', how='left')
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice'])

progress(f"Total entries: {len(df):,}")

# Calculate market implied probability
df['ImpliedProb'] = 1.0 / df['StartingPrice']

# Calculate form rate
df['FormRate'] = df.apply(
    lambda row: (row['RawWins'] / row['FormRaces']) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
    axis=1
)

# Normalize pace globally
pace_min = df['RawPaceAvg'].min()
pace_max = df['RawPaceAvg'].max()
df['PaceScore'] = (df['RawPaceAvg'] - pace_min) / (pace_max - pace_min)
df['PaceScore'] = df['PaceScore'].fillna(0)

# Model probability = weighted combo
df['ModelProb'] = (df['PaceScore'] * 0.7) + (df['FormRate'] * 0.3)

# Value = Model thinks dog is better than market
df['Edge'] = df['ModelProb'] - df['ImpliedProb']

days = 30

progress("\n" + "=" * 100)
progress("ANALYSIS BY ODDS BRACKET + MODEL EDGE")
progress("=" * 100)

# Test: Does the model identify value at different edge levels?
for min_edge in [-0.1, 0, 0.05, 0.1, 0.15, 0.2]:
    progress(f"\n--- Edge >= {min_edge:.2f} ---")
    
    edge_filter = df[df['Edge'] >= min_edge]
    
    for price_min, price_max in [(1.50, 2.50), (2.50, 4.00), (4.00, 6.00)]:
        bracket = edge_filter[(edge_filter['StartingPrice'] >= price_min) & (edge_filter['StartingPrice'] < price_max)]
        
        if len(bracket) < 10:
            continue
        
        wins = bracket['ActualWinner'].sum()
        bets = len(bracket)
        strike = wins / bets * 100
        avg_odds = bracket['StartingPrice'].mean()
        implied = 1 / avg_odds * 100  # Expected win rate from market
        profit = (wins * avg_odds) - bets
        roi = profit / bets * 100
        daily = bets / days
        
        # Is actual strike rate better than market expected?
        edge_vs_market = strike - implied
        
        marker = "<<<" if roi > 20 else ""
        progress(f"${price_min:.2f}-${price_max:.2f}: {bets:4} ({daily:.1f}/day) | Strike: {strike:.1f}% vs Market: {implied:.1f}% | Edge: {edge_vs_market:+.1f}% | ROI: {roi:+.1f}% {marker}")

progress("\n" + "=" * 100)
progress("FINDING SWEET SPOTS")
progress("=" * 100)

# Look for specific combinations that work
best_results = []

for min_edge in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    for min_form in [0, 0.2, 0.4, 0.6]:  # Minimum recent win rate
        for price_min, price_max in [(1.50, 3.00), (2.00, 4.00), (2.50, 4.00), (3.00, 5.00)]:
            
            filtered = df[
                (df['Edge'] >= min_edge) &
                (df['FormRate'] >= min_form) &
                (df['StartingPrice'] >= price_min) &
                (df['StartingPrice'] < price_max)
            ]
            
            if len(filtered) < 30:  # Need minimum sample
                continue
            
            wins = filtered['ActualWinner'].sum()
            bets = len(filtered)
            strike = wins / bets * 100
            avg_odds = filtered['StartingPrice'].mean()
            profit = (wins * avg_odds) - bets
            roi = profit / bets * 100
            daily = bets / days
            
            best_results.append({
                'config': f"Edge>={min_edge:.2f}, Form>={min_form:.1f}, ${price_min:.2f}-${price_max:.2f}",
                'bets': bets,
                'daily': daily,
                'wins': wins,
                'strike': strike,
                'roi': roi
            })

# Sort by ROI
results_df = pd.DataFrame(best_results)
results_df = results_df.sort_values('roi', ascending=False)

progress("\nTop 15 configurations by ROI (min 30 bets):")
for idx, row in results_df.head(15).iterrows():
    marker = "<<<" if row['roi'] > 15 and row['daily'] >= 3 else ""
    progress(f"{row['config']:<50} | {row['bets']:4} ({row['daily']:.1f}/day) | {row['strike']:.1f}% | ROI: {row['roi']:+.1f}% {marker}")

# Show configs with 3-10 bets/day
good_volume = results_df[(results_df['daily'] >= 3) & (results_df['daily'] <= 12)]
good_volume = good_volume.sort_values('roi', ascending=False)

progress("\nBest with 3-12 bets/day:")
for idx, row in good_volume.head(10).iterrows():
    progress(f"{row['config']:<50} | {row['bets']:4} ({row['daily']:.1f}/day) | {row['strike']:.1f}% | ROI: {row['roi']:+.1f}%")
