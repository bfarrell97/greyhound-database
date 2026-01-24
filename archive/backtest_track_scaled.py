"""
Track-scaled model: Adjust pace and form scores based on track difficulty
City/Major tracks = higher value for good results
Provincial/smaller tracks = lower value for results

70% Pace + 30% Form @ 0.80 threshold on $1.50-$3.00 range
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Track tier definitions
# NOTE: Removed NZ and TAS (Hobart, Launceston, Devonport) tracks
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

TRACK_WEIGHTS = {'metro': 1.0, 'provincial': 0.7, 'country': 0.3}

def get_track_tier_weight(track_name):
    if track_name in METRO_TRACKS:
        return TRACK_WEIGHTS['metro']
    elif track_name in PROVINCIAL_TRACKS:
        return TRACK_WEIGHTS['provincial']
    else:
        return TRACK_WEIGHTS['country']

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'provincial'
    else:
        return 'country'

progress("=" * 100)
progress("TRACK-SCALED MODEL: Metro=1.0x, Provincial=0.7x, Country=0.3x")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# First, identify tracks and their tiers
query_track_tiers = """
SELECT DISTINCT TrackName FROM Tracks
"""

progress("\nIdentifying track tiers...", indent=1)
tracks_df = pd.read_sql_query(query_track_tiers, conn)

# Add tier and weight columns
tracks_df['Tier'] = tracks_df['TrackName'].apply(get_track_tier)
tracks_df['TrackScaling'] = tracks_df['TrackName'].apply(get_track_tier_weight)

progress(f"Found {len(tracks_df)} unique tracks\n", indent=1)
progress("Track tier distribution:", indent=1)

for tier in ['metro', 'provincial', 'country']:
    tier_tracks = tracks_df[tracks_df['Tier'] == tier]
    progress(f"  {tier.upper():12} ({len(tier_tracks):2} tracks) - Weight: {TRACK_WEIGHTS[tier]:.1f}x", indent=2)
    for i, row in tier_tracks.iterrows():
        progress(f"    - {row['TrackName']}", indent=3)

# Get all unique race dates
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-10-01'
  AND rm.MeetingDate <= '2025-12-31'
ORDER BY RaceDate
"""

progress("Getting race dates...", indent=1)
dates_df = pd.read_sql_query(query_dates, conn)
race_dates = dates_df['RaceDate'].tolist()
progress(f"Found {len(race_dates)} unique race dates\n", indent=1)

all_results = []

progress("Processing races with track scaling...\n", indent=1)

for idx, race_date in enumerate(race_dates, 1):
    if idx % 15 == 0:
        progress(f"[{idx}/{len(race_dates)}] Processing {race_date}... ({len(all_results)} dogs so far)", indent=1)

    query = f"""
    WITH dog_pace_history AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            rm.MeetingDate,
            t.TrackName,
            (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
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
    
    dog_pace_avg AS (
        SELECT 
            GreyhoundID,
            AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg,
            COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as PacesUsed
        FROM dog_pace_history
        GROUP BY GreyhoundID
        HAVING PacesUsed >= 5
    ),
    
    dog_recent_form AS (
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
    
    dog_form_last_5 AS (
        SELECT 
            GreyhoundID,
            SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) as RecentWins,
            COUNT(*) as RecentRaces,
            ROUND(100.0 * SUM(CASE WHEN IsWinner = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as RecentWinRate
        FROM dog_recent_form
        WHERE RaceNum <= 5
        GROUP BY GreyhoundID
    ),
    
    todays_races AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            ge.StartingPrice,
            t.TrackName,
            rm.MeetingAvgBenchmarkLengths,
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
        tr.MeetingAvgBenchmarkLengths,
        dpa.HistoricalPaceAvg,
        COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
        tr.ActualWinner
    FROM todays_races tr
    LEFT JOIN dog_pace_avg dpa ON tr.GreyhoundID = dpa.GreyhoundID
    LEFT JOIN dog_form_last_5 dfl ON tr.GreyhoundID = dfl.GreyhoundID
    WHERE tr.StartingPrice >= 1.5 AND tr.StartingPrice <= 3.0
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
df = df.dropna(subset=['StartingPrice', 'HistoricalPaceAvg'])

progress(f"Total dog performances: {len(df)}\n", indent=1)

# Add track scaling to dataframe
df = df.merge(tracks_df[['TrackName', 'Tier', 'TrackScaling']], on='TrackName', how='left')
df['TrackScaling'] = df['TrackScaling'].fillna(0.3)  # Default to country weight if not found

# Normalize pace and form BEFORE applying track scaling
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()
if pace_max - pace_min > 0:
    df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
else:
    df['PaceScore'] = 0.5

df['FormScore'] = df['RecentWinRate'] / 100.0

# Apply track scaling to both pace and form
df['PaceScore_Scaled'] = df['PaceScore'] * df['TrackScaling']
df['FormScore_Scaled'] = df['FormScore'] * df['TrackScaling']

# Renormalize after scaling (so they stay in 0-1 range roughly)
df['PaceScore_Scaled'] = df['PaceScore_Scaled'].clip(0, 1)
df['FormScore_Scaled'] = df['FormScore_Scaled'].clip(0, 1)

# Calculate weighted score with scaled metrics
df['WeightedScore'] = (df['PaceScore_Scaled'] * 0.7) + (df['FormScore_Scaled'] * 0.3)

# Calculate implied probability
df['ImpliedProb'] = 1.0 / df['StartingPrice']

# Compare standard vs track-scaled model
progress("=" * 100)
progress("COMPARISON: Standard Model vs Track-Scaled Model")
progress("=" * 100)

# Standard model (no scaling)
df['WeightedScore_Standard'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
high_conf_standard = df[df['WeightedScore_Standard'] >= 0.80].copy()

# Track-scaled model
high_conf_scaled = df[df['WeightedScore'] >= 0.80].copy()

progress(f"\nStandard Model (70/30 @ 0.80):", indent=1)
if len(high_conf_standard) > 0:
    wins_std = high_conf_standard['ActualWinner'].sum()
    strike_std = (wins_std / len(high_conf_standard)) * 100
    roi_std = ((wins_std * high_conf_standard['StartingPrice'].mean()) - len(high_conf_standard)) / len(high_conf_standard) * 100
    progress(f"  Bets: {len(high_conf_standard):5} | Strike: {strike_std:5.1f}% | Avg Odds: ${high_conf_standard['StartingPrice'].mean():.2f} | ROI: {roi_std:+6.1f}%", indent=2)
else:
    progress(f"  Bets: 0 (no qualifying picks)", indent=2)

progress(f"\nTrack-Scaled Model (70/30 @ 0.80 with track scaling):", indent=1)
if len(high_conf_scaled) > 0:
    wins_sc = high_conf_scaled['ActualWinner'].sum()
    strike_sc = (wins_sc / len(high_conf_scaled)) * 100
    roi_sc = ((wins_sc * high_conf_scaled['StartingPrice'].mean()) - len(high_conf_scaled)) / len(high_conf_scaled) * 100
    progress(f"  Bets: {len(high_conf_scaled):5} | Strike: {strike_sc:5.1f}% | Avg Odds: ${high_conf_scaled['StartingPrice'].mean():.2f} | ROI: {roi_sc:+6.1f}%", indent=2)
else:
    progress(f"  Bets: 0 (no qualifying picks)", indent=2)

# Detailed breakdown by track tier
progress(f"\n" + "=" * 100)
progress("HIGH CONFIDENCE PICKS BY TRACK TIER (Track-Scaled Model)")
progress("=" * 100)

for tier in ['metro', 'provincial', 'country']:
    tier_data = high_conf_scaled[high_conf_scaled['Tier'] == tier]
    if len(tier_data) == 0:
        continue
        
    wins = tier_data['ActualWinner'].sum()
    strike = (wins / len(tier_data)) * 100
    avg_odds = tier_data['StartingPrice'].mean()
    scaling = TRACK_WEIGHTS[tier]
    roi = ((wins * avg_odds) - len(tier_data)) / len(tier_data) * 100
    
    progress(f"\n{tier.upper()} (Weight: {scaling:.1f}x):", indent=1)
    progress(f"  Bets: {len(tier_data)} | Wins: {wins} | Strike: {strike:.1f}% | Avg Odds: ${avg_odds:.2f} | ROI: {roi:+.1f}%", indent=2)

progress("\n" + "=" * 100)

conn.close()
