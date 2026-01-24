"""
Track-scaled pace & form model: 
Scale the actual performance metrics by track difficulty
Metro performance = 4x multiplier
Provincial performance = 2x multiplier  
Country performance = 1x multiplier

Apply scaling to both pace (FinishTimeBenchmarkLengths) and form (wins)
Then calculate 70/30 weighted score @ 0.80 on $1.50-$3.00 range
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Track tier definitions (exclude NZ and TAS)
METRO_TRACKS = {
    'Wentworth Park', 'Albion Park', 'Angle Park',
    'Sandown Park', 'The Meadows', 'Cannington'
}

PROVINCIAL_TRACKS = {
    'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli',
    'Dapto', 'Maitland', 'Goulburn', 'Ipswich', 'Q Straight',
    'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
    'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'
}

# Scaling multipliers: how much track tier affects the quality of performance
TRACK_SCALING = {'metro': 4.0, 'provincial': 2.0, 'country': 1.0}

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'provincial'
    else:
        return 'country'

progress("=" * 100)
progress("TRACK-SCALED PACE & FORM MODEL")
progress("=" * 100)
progress("Scale: Metro=4x, Provincial=2x, Country=1x")
progress("Applies to both pace benchmarks and form (wins)\n")

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all unique race dates (Sept, Oct, Nov)
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-09-01'
  AND rm.MeetingDate < '2025-12-01'
ORDER BY RaceDate
"""

progress("Getting race dates...", indent=1)
dates_df = pd.read_sql_query(query_dates, conn)
race_dates = dates_df['RaceDate'].tolist()
progress(f"Found {len(race_dates)} unique race dates\n", indent=1)

all_results = []

progress("Processing races with track-scaled metrics...\n", indent=1)

for idx, race_date in enumerate(race_dates, 1):
    if idx % 15 == 0:
        progress(f"[{idx}/{len(race_dates)}] Processing {race_date}... ({len(all_results)} dogs so far)", indent=1)

    # Get scaled pace metrics BEFORE this date
    query = f"""
    WITH dog_pace_history_raw AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            rm.MeetingDate,
            t.TrackName,
            ge.FinishTimeBenchmarkLengths,
            rm.MeetingAvgBenchmarkLengths,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < DATE('{race_date}')
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
    ),
    
    dog_pace_scaled AS (
        SELECT 
            GreyhoundID,
            AVG(
                CASE 
                    WHEN RaceNum <= 5 THEN 
                        (FinishTimeBenchmarkLengths + MeetingAvgBenchmarkLengths)
                    ELSE NULL
                END
            ) as RawPaceAvg,
            COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
        FROM dog_pace_history_raw
        GROUP BY GreyhoundID
        HAVING PacesUsed >= 5
    ),
    
    dog_recent_form_raw AS (
        SELECT 
            ge.GreyhoundID,
            t.TrackName,
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
            rm.MeetingDate,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < DATE('{race_date}')
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
    ),
    
    dog_form_scaled AS (
        SELECT 
            GreyhoundID,
            SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
            COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
        FROM dog_recent_form_raw
        GROUP BY GreyhoundID
    ),
    
    todays_races AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            ge.StartingPrice,
            t.TrackName,
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE DATE(rm.MeetingDate) = DATE('{race_date}')
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND ge.StartingPrice IS NOT NULL
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
    )
    
    SELECT 
        tr.GreyhoundID,
        tr.GreyhoundName,
        tr.StartingPrice,
        tr.TrackName,
        dps.RawPaceAvg,
        dfs.RawWins,
        dfs.FormRaces,
        tr.ActualWinner
    FROM todays_races tr
    LEFT JOIN dog_pace_scaled dps ON tr.GreyhoundID = dps.GreyhoundID
    LEFT JOIN dog_form_scaled dfs ON tr.GreyhoundID = dfs.GreyhoundID
    WHERE tr.StartingPrice >= 1.5 AND tr.StartingPrice <= 5.0
    """
    
    df = pd.read_sql_query(query, conn)
    
    if len(df) > 0:
        df['RaceDate'] = race_date
        all_results.append(df)

progress(f"\nProcessed {len(all_results)} race dates\n", indent=1)

if len(all_results) == 0:
    progress("No data found!")
    conn.close()
    exit()

# Combine all results
df = pd.concat(all_results, ignore_index=True)

# Convert to numeric
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df = df.dropna(subset=['StartingPrice'])

progress(f"Total dog performances: {len(df)}\n", indent=1)

# Add track tier
df['TrackTier'] = df['TrackName'].apply(get_track_tier)

# Apply track scaling to raw metrics
df['ScaledPace'] = df.apply(
    lambda row: row['RawPaceAvg'] * TRACK_SCALING[row['TrackTier']] if pd.notna(row['RawPaceAvg']) else 0,
    axis=1
)

df['ScaledWins'] = df.apply(
    lambda row: row['RawWins'] * TRACK_SCALING[row['TrackTier']] if pd.notna(row['RawWins']) else 0,
    axis=1
)

# Normalize scaled metrics to 0-1 scores
pace_min = df[df['ScaledPace'] > 0]['ScaledPace'].min()
pace_max = df['ScaledPace'].max()

if pace_max - pace_min > 0:
    df['PaceScore'] = (df['ScaledPace'] - pace_min) / (pace_max - pace_min)
else:
    df['PaceScore'] = 0.5

# Form score: wins per race
df['FormRateScaled'] = df.apply(
    lambda row: (row['ScaledWins'] / row['FormRaces'] * 100) if row['FormRaces'] > 0 else 0,
    axis=1
)
df['FormScore'] = df['FormRateScaled'] / 100.0

# Calculate weighted score
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Calculate implied probability
df['ImpliedProb'] = 1.0 / df['StartingPrice']

# Filter to 0.80+ confidence
high_conf = df[df['WeightedScore'] >= 0.80].copy()

progress("=" * 100)
progress("TRACK-SCALED MODEL RESULTS (70/30 @ 0.80)")
progress("=" * 100)

total_bets = len(high_conf)
total_wins = high_conf['ActualWinner'].sum()
strike_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
avg_odds = high_conf['StartingPrice'].mean()
roi = ((total_wins * avg_odds) - total_bets) / total_bets * 100 if total_bets > 0 else 0

progress(f"\nTotal Bets: {total_bets}", indent=1)
progress(f"Total Wins: {total_wins}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# Breakdown by track tier
progress(f"\n" + "=" * 100)
progress("BREAKDOWN BY TRACK TIER")
progress("=" * 100)

for tier in ['metro', 'provincial', 'country']:
    tier_data = high_conf[high_conf['TrackTier'] == tier]
    if len(tier_data) == 0:
        continue
        
    wins = tier_data['ActualWinner'].sum()
    strike = (wins / len(tier_data)) * 100
    avg_odds_tier = tier_data['StartingPrice'].mean()
    roi_tier = ((wins * avg_odds_tier) - len(tier_data)) / len(tier_data) * 100
    
    progress(f"\n{tier.upper()} (Scale: {TRACK_SCALING[tier]:.1f}x):", indent=1)
    progress(f"  Bets: {len(tier_data)} | Wins: {wins} | Strike: {strike:.1f}% | Avg Odds: ${avg_odds_tier:.2f} | ROI: {roi_tier:+.1f}%", indent=2)

# Breakdown by price brackets
progress(f"\n" + "=" * 100)
progress("BREAKDOWN BY PRICE BRACKETS")
progress("=" * 100)

price_brackets = [
    (1.50, 2.00, "$1.50-$2.00"),
    (2.00, 2.50, "$2.00-$2.50"),
    (2.50, 3.00, "$2.50-$3.00"),
    (3.00, 4.00, "$3.00-$4.00"),
    (4.00, 5.00, "$4.00-$5.00"),
]

for min_price, max_price, label in price_brackets:
    bracket_data = high_conf[(high_conf['StartingPrice'] >= min_price) & (high_conf['StartingPrice'] < max_price)]
    if len(bracket_data) == 0:
        continue
    
    wins = bracket_data['ActualWinner'].sum()
    strike = (wins / len(bracket_data)) * 100
    avg_odds_bracket = bracket_data['StartingPrice'].mean()
    roi_bracket = ((wins * avg_odds_bracket) - len(bracket_data)) / len(bracket_data) * 100
    
    progress(f"\n{label}:", indent=1)
    progress(f"  Bets: {len(bracket_data)} | Wins: {wins} | Strike: {strike:.1f}% | Avg Odds: ${avg_odds_bracket:.2f} | ROI: {roi_bracket:+.1f}%", indent=2)

progress("\n" + "=" * 100)

conn.close()
