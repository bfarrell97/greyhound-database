"""
Fast Weekly Backtest - Updates pace AND form data each week
6 months (June-Nov 2025), $2.50-$4.00 range
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Track tier definitions
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

TRACK_SCALING = {'metro': 4.0, 'provincial': 2.0, 'country': 1.0}

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'provincial'
    else:
        return 'country'

progress("=" * 100)
progress("WEEKLY BACKTEST - ROLLING PACE & FORM")
progress("=" * 100)
progress("Period: June-November 2025 (6 months)")
progress("Price Range: $2.50-$4.00")
progress("Updates pace & form weekly\n")

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Generate weekly cutoff dates (Sundays from June to Nov)
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 12, 1)

# Get all weeks
weeks = []
current = start_date
while current < end_date:
    week_end = current + timedelta(days=6)
    if week_end > end_date:
        week_end = end_date
    weeks.append((current.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d')))
    current = week_end + timedelta(days=1)

progress(f"Processing {len(weeks)} weeks...\n", indent=1)

all_bets = []

for idx, (week_start, week_end) in enumerate(weeks, 1):
    progress(f"[{idx}/{len(weeks)}] Week {week_start} to {week_end}...", indent=1)
    
    # Get pace data BEFORE this week starts
    pace_query = f"""
    WITH dog_pace_history_raw AS (
        SELECT 
            ge.GreyhoundID,
            rm.MeetingDate,
            t.TrackName,
            (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalBench,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < '{week_start}'
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
    """
    
    pace_df = pd.read_sql_query(pace_query, conn)
    
    # Get form data BEFORE this week starts
    form_query = f"""
    WITH dog_form_raw AS (
        SELECT 
            ge.GreyhoundID,
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
            rm.MeetingDate,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < '{week_start}'
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
    )
    SELECT 
        GreyhoundID,
        SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
        COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
    FROM dog_form_raw
    GROUP BY GreyhoundID
    """
    
    form_df = pd.read_sql_query(form_query, conn)
    
    # Get races in this week
    races_query = f"""
    SELECT 
        ge.GreyhoundID,
        g.GreyhoundName,
        ge.StartingPrice,
        t.TrackName,
        DATE(rm.MeetingDate) as RaceDate,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '{week_start}'
      AND rm.MeetingDate <= '{week_end}'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
      AND ge.StartingPrice >= 2.5 AND ge.StartingPrice <= 4.0
    """
    
    df = pd.read_sql_query(races_query, conn)
    
    if len(df) == 0:
        continue
    
    # Merge pace and form
    df = df.merge(pace_df, on='GreyhoundID', how='left')
    df = df.merge(form_df, on='GreyhoundID', how='left')
    
    # Only keep dogs with valid pace data
    df = df.dropna(subset=['RawPaceAvg'])
    
    if len(df) == 0:
        continue
    
    # Convert to numeric and filter by price range
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df = df.dropna(subset=['StartingPrice'])
    df = df[(df['StartingPrice'] >= 2.5) & (df['StartingPrice'] <= 4.0)]
    
    if len(df) == 0:
        continue
    
    # Add track tier
    df['TrackTier'] = df['TrackName'].apply(get_track_tier)
    
    # Apply track scaling to PACE only (form is already a rate)
    df['ScaledPace'] = df.apply(
        lambda row: row['RawPaceAvg'] * TRACK_SCALING[row['TrackTier']] if pd.notna(row['RawPaceAvg']) else 0,
        axis=1
    )
    
    # Calculate form rate (wins / races as percentage, capped at 100%)
    df['FormRate'] = df.apply(
        lambda row: min((row['RawWins'] / row['FormRaces'] * 100), 100) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
        axis=1
    )
    
    # Normalize pace scores (within this week's candidates)
    pace_min = df[df['ScaledPace'] > 0]['ScaledPace'].min() if len(df[df['ScaledPace'] > 0]) > 0 else 0
    pace_max = df['ScaledPace'].max()
    
    if pace_max - pace_min > 0:
        df['PaceScore'] = (df['ScaledPace'] - pace_min) / (pace_max - pace_min)
    else:
        df['PaceScore'] = 0.5
    
    # Form score is already 0-100, just convert to 0-1
    df['FormScore'] = df['FormRate'] / 100.0
    
    # Weighted score
    df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)
    
    # Filter to high confidence
    high_conf = df[df['WeightedScore'] >= 0.80].copy()
    
    if len(high_conf) > 0:
        high_conf['Profit'] = high_conf.apply(lambda row: row['StartingPrice'] - 1 if row['ActualWinner'] == 1 else -1, axis=1)
        all_bets.append(high_conf)

conn.close()

if len(all_bets) == 0:
    progress("No bets generated!")
    exit()

# Combine all bets
results = pd.concat(all_bets, ignore_index=True)
results = results.sort_values('RaceDate')

progress(f"\n{'='*100}")
progress("RESULTS: ROLLING WEEKLY PACE & FORM (70/30 @ 0.80)")
progress(f"{'='*100}")

total_bets = len(results)
total_wins = results['ActualWinner'].sum()
strike_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
avg_odds = results['StartingPrice'].mean()
total_profit = results['Profit'].sum()
roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

progress(f"\nTotal Bets: {total_bets}", indent=1)
progress(f"Total Wins: {total_wins}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Average Odds: ${avg_odds:.2f}", indent=1)
progress(f"Total Profit: {total_profit:+.1f} units", indent=1)
progress(f"ROI: {roi:+.1f}%", indent=1)

# Monthly breakdown
progress(f"\n{'='*100}")
progress("BREAKDOWN BY MONTH")
progress(f"{'='*100}")

results['Month'] = pd.to_datetime(results['RaceDate']).dt.strftime('%Y-%m')

for month in sorted(results['Month'].unique()):
    month_data = results[results['Month'] == month]
    wins = month_data['ActualWinner'].sum()
    strike = (wins / len(month_data)) * 100 if len(month_data) > 0 else 0
    avg_odds_month = month_data['StartingPrice'].mean()
    profit = month_data['Profit'].sum()
    roi_month = (profit / len(month_data)) * 100 if len(month_data) > 0 else 0
    
    progress(f"\n{month}:", indent=1)
    progress(f"  Bets: {len(month_data)} | Wins: {wins} | Strike: {strike:.1f}% | Avg Odds: ${avg_odds_month:.2f} | Profit: {profit:+.1f} | ROI: {roi_month:+.1f}%", indent=2)

# Export CSV
csv_file = 'backtest_weekly_all_bets.csv'
export_cols = ['RaceDate', 'GreyhoundName', 'TrackName', 'TrackTier', 'StartingPrice', 
               'WeightedScore', 'PaceScore', 'FormScore', 'ActualWinner', 'Profit']
results[export_cols].to_csv(csv_file, index=False)

progress(f"\n{'='*100}")
progress(f"Exported {len(results)} bets to {csv_file}")
progress(f"{'='*100}")
