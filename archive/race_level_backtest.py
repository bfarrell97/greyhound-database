"""
Race-Level Normalization Backtest
Instead of normalizing globally, normalize within each race
This identifies the BEST dog in each race based on pace/form
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("RACE-LEVEL NORMALIZATION - Pick best dog per race")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get pace data (before November)
progress("Loading pace data...", indent=1)
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

# Get form data
progress("Loading form data...", indent=1)
form_df = pd.read_sql_query("""
WITH dog_form_raw AS (
    SELECT 
        ge.GreyhoundID,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        rm.MeetingDate,
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

# Get November races WITH RaceID for grouping
progress("Loading November races...", indent=1)
races_df = pd.read_sql_query("""
SELECT 
    ge.GreyhoundID,
    g.GreyhoundName,
    r.RaceID,
    ge.StartingPrice,
    t.TrackName,
    DATE(rm.MeetingDate) as RaceDate,
    (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
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

progress(f"Loaded {len(races_df):,} race entries in {races_df['RaceID'].nunique()} races\n", indent=1)

# Merge
df = races_df.merge(pace_df, on='GreyhoundID', how='left')
df = df.merge(form_df, on='GreyhoundID', how='left')

# Clean
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice'])

# Calculate form rate
df['FormRate'] = df.apply(
    lambda row: (row['RawWins'] / row['FormRaces'] * 100) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
    axis=1
)

# RACE-LEVEL NORMALIZATION
# For each race, normalize pace and form scores within that race
progress("Normalizing within each race...", indent=1)

def normalize_within_race(group):
    group = group.copy()
    # Pace normalization (higher is better - more lengths ahead of benchmark)
    if group['RawPaceAvg'].notna().sum() > 0:
        pace_min = group['RawPaceAvg'].min()
        pace_max = group['RawPaceAvg'].max()
        if pace_max - pace_min > 0:
            group['PaceRank'] = (group['RawPaceAvg'] - pace_min) / (pace_max - pace_min)
        else:
            group['PaceRank'] = 0.5
    else:
        group['PaceRank'] = 0.0
    
    # Form normalization
    form_min = group['FormRate'].min()
    form_max = group['FormRate'].max()
    if form_max - form_min > 0:
        group['FormRank'] = (group['FormRate'] - form_min) / (form_max - form_min)
    else:
        group['FormRank'] = 0.5
    
    return group

df = df.groupby('RaceID', group_keys=False).apply(normalize_within_race).reset_index(drop=True)

# Weighted score (within race context)
df['RaceScore'] = (df['PaceRank'] * 0.7) + (df['FormRank'] * 0.3)

# For each race, identify the top-scored dog(s)
progress("Testing selection strategies...\n", indent=1)

days = 30

progress("=" * 100)
progress("STRATEGY 1: Bet on highest-scored dog in each race (if in price range)")
progress("=" * 100)

for price_min, price_max in [(1.50, 3.00), (1.50, 4.00), (2.00, 4.00), (2.50, 4.00), (2.00, 5.00)]:
    # Get best dog per race
    price_filtered = df[(df['StartingPrice'] >= price_min) & (df['StartingPrice'] <= price_max)].copy()
    
    if len(price_filtered) == 0:
        continue
    
    # Drop rows with NaN RaceScore
    price_filtered = price_filtered.dropna(subset=['RaceScore'])
    
    if len(price_filtered) == 0:
        continue
    
    # Group by race, get top scorer
    idx = price_filtered.groupby('RaceID')['RaceScore'].idxmax()
    top_per_race = price_filtered.loc[idx]
    
    wins = top_per_race['ActualWinner'].sum()
    bets = len(top_per_race)
    strike = wins / bets * 100 if bets > 0 else 0
    avg_odds = top_per_race['StartingPrice'].mean()
    profit = (wins * avg_odds) - bets
    roi = profit / bets * 100 if bets > 0 else 0
    daily = bets / days
    
    marker = "<<<" if roi > 20 and daily >= 3 else ""
    progress(f"${price_min:.2f}-${price_max:.2f}: {bets:4} bets ({daily:.1f}/day) | {wins} wins | {strike:.1f}% | ROI: {roi:+.1f}% {marker}")

progress("\n" + "=" * 100)
progress("STRATEGY 2: Bet on top dog IF score > threshold AND they're also the favourite")
progress("=" * 100)

for min_score in [0.7, 0.8, 0.9]:
    for price_min, price_max in [(1.50, 3.00), (2.00, 4.00), (2.00, 5.00)]:
        price_filtered = df[(df['StartingPrice'] >= price_min) & (df['StartingPrice'] <= price_max)]
        
        if len(price_filtered) == 0:
            continue
        
        # Get races where our top-scored dog also has the lowest odds (is favourite)
        def is_fav_and_top(group):
            if len(group) == 0:
                return pd.DataFrame()
            top_scorer = group.loc[group['RaceScore'].idxmax()]
            min_odds_dog = group.loc[group['StartingPrice'].idxmin()]
            
            # Top scorer must also be favourite (or close to it)
            if top_scorer.name == min_odds_dog.name and top_scorer['RaceScore'] >= min_score:
                return pd.DataFrame([top_scorer])
            return pd.DataFrame()
        
        selections = price_filtered.groupby('RaceID').apply(is_fav_and_top).reset_index(drop=True)
        
        if len(selections) == 0:
            continue
        
        wins = selections['ActualWinner'].sum()
        bets = len(selections)
        strike = wins / bets * 100
        avg_odds = selections['StartingPrice'].mean()
        profit = (wins * avg_odds) - bets
        roi = profit / bets * 100
        daily = bets / days
        
        marker = "<<<" if roi > 20 and daily >= 3 else ""
        progress(f"Score>{min_score:.1f} + Fav ${price_min:.2f}-${price_max:.2f}: {bets:4} bets ({daily:.1f}/day) | {wins} wins | {strike:.1f}% | ROI: {roi:+.1f}% {marker}")

progress("\n" + "=" * 100)
progress("STRATEGY 3: Value betting - top scored dog at higher odds than implied")
progress("=" * 100)

for min_edge in [0.1, 0.2, 0.3]:  # How much higher RaceScore vs implied probability
    for price_min, price_max in [(2.00, 5.00), (2.50, 5.00), (3.00, 6.00)]:
        price_filtered = df[(df['StartingPrice'] >= price_min) & (df['StartingPrice'] <= price_max)]
        
        if len(price_filtered) == 0:
            continue
        
        # Implied probability from odds
        price_filtered = price_filtered.copy()
        price_filtered['ImpliedProb'] = 1.0 / price_filtered['StartingPrice']
        
        # Value = RaceScore > ImpliedProb + edge
        value_bets = price_filtered[price_filtered['RaceScore'] > price_filtered['ImpliedProb'] + min_edge]
        
        # Take only top scorer per race
        if len(value_bets) == 0:
            continue
            
        value_bets = value_bets.loc[value_bets.groupby('RaceID')['RaceScore'].idxmax()]
        
        wins = value_bets['ActualWinner'].sum()
        bets = len(value_bets)
        strike = wins / bets * 100
        avg_odds = value_bets['StartingPrice'].mean()
        profit = (wins * avg_odds) - bets
        roi = profit / bets * 100
        daily = bets / days
        
        marker = "<<<" if roi > 20 and daily >= 3 else ""
        progress(f"Edge>{min_edge:.1f} ${price_min:.2f}-${price_max:.2f}: {bets:4} bets ({daily:.1f}/day) | {wins} wins | {strike:.1f}% | ROI: {roi:+.1f}% {marker}")
