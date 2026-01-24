"""
Build simple track-specific selection rules
Focus on tracks where favorites outperform (Nowra 58.1%, Q Straight 57.2%, Devonport 60%)
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'

print("="*80)
print("TRACK-SPECIFIC STRATEGY: Simple Rules on High-Win Tracks")
print("="*80)

conn = sqlite3.connect(DB_PATH)

# Focus on highest-performing tracks for $1.50-$2.00 favorites
high_win_tracks = ['Nowra', 'Q Straight', 'Dport @ HOB']

print(f"\nAnalyzing tracks: {high_win_tracks}")

# Load data
query = """
SELECT
    g.GreyhoundName, t.TrackName, r.Distance, ge.Box, ge.Weight, 
    ge.Position, ge.StartingPrice, ge.EarlySpeed, ge.Rating,
    ge.Form, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND t.TrackName IN (?, ?, ?)
ORDER BY rm.MeetingDate
"""

df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE) + tuple(high_win_tracks))
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['EarlySpeed'] = pd.to_numeric(df['EarlySpeed'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

print(f"\nLoaded {len(df):,} races on selected tracks")

# Filter to $1.50-$2.00
df = df[(df['StartingPrice'] >= 1.5) & (df['StartingPrice'] < 2.0)]
print(f"After filtering to $1.50-$2.00: {len(df):,} races")
print(f"Overall win rate: {df['IsWinner'].mean()*100:.1f}%")

# Strategy 1: Simple box + early speed
print("\n" + "="*80)
print("STRATEGY 1: Box Position + Early Speed")
print("="*80)

print(f"EarlySpeed values: min={df['EarlySpeed'].min()}, max={df['EarlySpeed'].max()}, "
      f"non-null={df['EarlySpeed'].notna().sum()}/{len(df)}")
print(f"Rating values: min={df['Rating'].min()}, max={df['Rating'].max()}, "
      f"non-null={df['Rating'].notna().sum()}/{len(df)}")

for box in [1, 2, 3]:
    strategy_df = df[df['Box'] <= box]
    if len(strategy_df) > 20:
        wins = strategy_df['IsWinner'].sum()
        win_rate = (wins / len(strategy_df)) * 100
        
        # Calculate ROI with flat 2% staking
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy_df) * bankroll * stake_pct
        returns = (strategy_df[strategy_df['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100
        
        print(f"Box <= {box}: {len(strategy_df):>4} bets, {win_rate:>5.1f}% strike, ROI {roi:>6.2f}%")

# Strategy 2: Weight-based
print("\n" + "="*80)
print("STRATEGY 2: Heavy Dogs (33+kg) at Distance >= 400m")
print("="*80)

for dist in [400, 350, 300]:
    strategy_df = df[(df['Distance'] >= dist) & (df['Weight'] >= 33)]
    if len(strategy_df) > 20:
        wins = strategy_df['IsWinner'].sum()
        win_rate = (wins / len(strategy_df)) * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(strategy_df) * bankroll * stake_pct
        returns = (strategy_df[strategy_df['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        pnl = returns - total_staked
        roi = (pnl / total_staked) * 100
        
        print(f"Distance >= {dist}m, Weight >= 33kg: {len(strategy_df):>4} bets, "
              f"{win_rate:>5.1f}% strike, ROI {roi:>6.2f}%")

# Strategy 3: Rating-based
print("\n" + "="*80)
print("STRATEGY 3: High Rating (50+)")
print("="*80)

if df['Rating'].notna().sum() > 0:
    for min_rating in [40, 50, 60]:
        strategy_df = df[df['Rating'] >= min_rating]
        if len(strategy_df) > 20:
            wins = strategy_df['IsWinner'].sum()
            win_rate = (wins / len(strategy_df)) * 100
            
            stake_pct = 0.02
            bankroll = 1000
            total_staked = len(strategy_df) * bankroll * stake_pct
            returns = (strategy_df[strategy_df['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
            pnl = returns - total_staked
            roi = (pnl / total_staked) * 100
            
            print(f"Rating >= {min_rating}: {len(strategy_df):>4} bets, "
                  f"{win_rate:>5.1f}% strike, ROI {roi:>6.2f}%")
else:
    print("No Rating data available")

# Strategy 4: Combination
print("\n" + "="*80)
print("STRATEGY 4: Box 1-2 + Rating 50+ (Best combo)")
print("="*80)

strategy_df = df[(df['Box'] <= 2) & (df['Rating'] >= 50)]
if len(strategy_df) > 20:
    wins = strategy_df['IsWinner'].sum()
    win_rate = (wins / len(strategy_df)) * 100
    
    stake_pct = 0.02
    bankroll = 1000
    total_staked = len(strategy_df) * bankroll * stake_pct
    returns = (strategy_df[strategy_df['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
    pnl = returns - total_staked
    roi = (pnl / total_staked) * 100
    
    print(f"Box <= 2, Rating >= 50: {len(strategy_df):>4} bets, "
          f"{win_rate:>5.1f}% strike, ROI {roi:>6.2f}%")
    
    # Show sample bets
    print(f"\nSample bets (first 10):")
    print(f"{'Dog':<20} {'Track':<15} {'Odds':<8} {'Box':<4} {'Rating':<8} {'Result':<8}")
    print("-"*80)
    for _, row in strategy_df.head(10).iterrows():
        result = "WIN" if row['IsWinner'] == 1 else "LOSE"
        print(f"{row['GreyhoundName']:<20} {row['TrackName']:<15} {row['StartingPrice']:<8.2f} "
              f"{int(row['Box']):<4} {row['Rating']:<8.0f} {result:<8}")

conn.close()

print("\n" + "="*80)
print("FINDINGS")
print("="*80)
print("""
This tests simple, explainable rules on the high-win tracks.
If any strategy beats 61% strike rate (the break-even for $1.50-$2.00),
we might have found something real.

Rating and EarlySpeed are direct quality indicators from the racing form,
not derived features, so they're less likely to be fully priced in.
""")
