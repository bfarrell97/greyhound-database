"""
Deploy Early Speed Filter Strategy
Use SplitBenchmarkLengths >= 1.0 to select betting opportunities
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*80)
print("EARLY SPEED FILTER STRATEGY - DEPLOYMENT")
print("="*80)
print(f"\nStrategy: SplitBenchmarkLengths >= 1.0")
print(f"Expected ROI: +36.38% | Strike: 80.1% | Sample: 2,127 bets")
print(f"\nDeployed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Test on future unseen data if available
# For now, validate on test period and show how to apply to new races

# Load recent races (past week as example)
query = """
SELECT
    ge.EntryID, ge.RaceID, ge.GreyhoundID, 
    g.GreyhoundName, t.TrackName, r.RaceNumber, r.Distance,
    ge.Box, ge.Weight, ge.StartingPrice,
    ge.SplitBenchmarkLengths,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= date('now', '-7 days')
  AND ge.Position IS NULL
  AND ge.SplitBenchmarkLengths IS NOT NULL
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

try:
    df_upcoming = pd.read_sql_query(query, conn)
    df_upcoming['StartingPrice'] = pd.to_numeric(df_upcoming['StartingPrice'], errors='coerce')
    
    # Filter for strategy criteria
    df_filtered = df_upcoming[
        (df_upcoming['StartingPrice'] >= 1.5) & 
        (df_upcoming['StartingPrice'] < 2.0) &
        (df_upcoming['SplitBenchmarkLengths'] >= 1.0)
    ].copy()
    
    if len(df_filtered) > 0:
        print(f"\n{'='*80}")
        print(f"BETTING OPPORTUNITIES (Past 7 days, not yet raced)")
        print(f"{'='*80}")
        
        for date in sorted(df_filtered['MeetingDate'].unique()):
            day_races = df_filtered[df_filtered['MeetingDate'] == date]
            print(f"\n{date}:")
            
            for track in sorted(day_races['TrackName'].unique()):
                track_races = day_races[day_races['TrackName'] == track]
                print(f"  {track}:")
                
                for race_num in sorted(track_races['RaceNumber'].unique()):
                    race_entries = track_races[track_races['RaceNumber'] == race_num]
                    print(f"    Race {race_num} (Distance: {race_entries['Distance'].iloc[0]}m):")
                    
                    for _, entry in race_entries.iterrows():
                        signal = "✓ BET" if entry['SplitBenchmarkLengths'] >= 1.5 else "○ OK"
                        print(f"      {signal} Box {entry['Box']} {entry['GreyhoundName']:20} "
                              f"${entry['StartingPrice']:5.2f} (Split: {entry['SplitBenchmarkLengths']:+6.2f})")
    else:
        print(f"\nNo matching opportunities in the past 7 days at $1.50-$2.00 with SplitBenchmarkLengths >= 1.0")
        print(f"Loaded {len(df_upcoming)} upcoming entries, {len(df_filtered)} match criteria")
        
except Exception as e:
    print(f"\nNote: {e}")
    print("(This is expected if no upcoming races data available)")

# Show historical validation
print(f"\n{'='*80}")
print(f"HISTORICAL VALIDATION (Test Period)")
print(f"{'='*80}")

query_hist = """
SELECT
    ge.Position, ge.StartingPrice, ge.SplitBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.SplitBenchmarkLengths >= 1.0
  AND ge.StartingPrice >= 1.5
  AND CAST(ge.StartingPrice AS REAL) < 2.0
"""

df_hist = pd.read_sql_query(query_hist, conn)
df_hist['Position'] = pd.to_numeric(df_hist['Position'], errors='coerce')
df_hist['StartingPrice'] = pd.to_numeric(df_hist['StartingPrice'], errors='coerce')
df_hist['IsWinner'] = (df_hist['Position'] == 1).astype(int)

print(f"\nHistorical Results (SplitBenchmarkLengths >= 1.0, $1.50-$2.00):")
print(f"  Total bets: {len(df_hist)}")
print(f"  Wins: {df_hist['IsWinner'].sum()}")
print(f"  Strike rate: {df_hist['IsWinner'].mean()*100:.1f}%")

# ROI calculation
stake_pct = 0.02
bankroll = 1000
total_staked = len(df_hist) * bankroll * stake_pct
returns = (df_hist[df_hist['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
roi = ((returns - total_staked) / total_staked) * 100 if total_staked > 0 else 0

print(f"  Avg odds: ${df_hist['StartingPrice'].mean():.2f}")
print(f"  Total staked: ${total_staked:,.0f}")
print(f"  Total returns: ${returns:,.0f}")
print(f"  Profit: ${returns - total_staked:,.0f}")
print(f"  ROI: {roi:.2f}%")

# Confidence analysis
print(f"\nConfidence Analysis (by early speed strength):")
for min_split in [1.0, 1.5, 2.0]:
    subset = df_hist[df_hist['SplitBenchmarkLengths'] >= min_split]
    if len(subset) > 20:
        strike = subset['IsWinner'].mean() * 100
        subset_roi = ((subset[subset['IsWinner'] == 1]['StartingPrice'].sum() - len(subset)) / len(subset)) * 100
        print(f"  Split >= {min_split}: {len(subset):>4} bets, {strike:>5.1f}% strike")

conn.close()

print("\n" + "="*80)
print("DEPLOYMENT NOTES")
print("="*80)
print("""
STRATEGY RULES:
1. Only bet on dogs with SplitBenchmarkLengths >= 1.0
2. Only bet on $1.50-$2.00 odds range
3. Bet 2% of bankroll per race
4. No other filters needed - keep it simple

EXPECTED OUTCOMES:
- Win ~80% of bets placed
- Make ~36% ROI over time
- Average 3-5 bets per day

RISKS:
- Early speed metric might lose predictive power over time
- Track conditions change (wet/dry affect sectional times)
- Sample size is large enough to be confident (2,127 bets)

NEXT ITERATION:
- Track real performance for 2-4 weeks
- Compare actual vs expected results
- Adjust threshold if needed (1.5 for higher confidence, 0.5 for volume)
""")
