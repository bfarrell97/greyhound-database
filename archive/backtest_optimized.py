"""
Optimized Weekly Backtest
Goal: 3-10 bets/day, ~30% ROI
Based on analysis: Use threshold 0.60, price $2.50-$5.00
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Configuration - ADJUSTABLE PARAMETERS
THRESHOLD = 0.60  # Lower threshold for more bets
PRICE_MIN = 2.50
PRICE_MAX = 5.00
PACE_WEIGHT = 0.70
FORM_WEIGHT = 0.30
MIN_PACE_RACES = 3  # Reduced from 5 to include more dogs

progress("=" * 100)
progress("OPTIMIZED WEEKLY BACKTEST")
progress("=" * 100)
progress(f"Threshold: {THRESHOLD} | Price: ${PRICE_MIN}-${PRICE_MAX}")
progress(f"Weights: Pace {PACE_WEIGHT*100:.0f}% / Form {FORM_WEIGHT*100:.0f}%")
progress(f"Min pace races: {MIN_PACE_RACES}")
progress("Period: June-November 2025 (6 months)\n")

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Generate weekly cutoff dates
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 12, 1)

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
    if idx % 5 == 0:
        progress(f"[{idx}/{len(weeks)}] Week {week_start}...", indent=1)
    
    # Get pace data BEFORE this week
    pace_query = f"""
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
    HAVING PacesUsed >= {MIN_PACE_RACES}
    """
    
    pace_df = pd.read_sql_query(pace_query, conn)
    
    # Get form data BEFORE this week
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
        WHERE ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < '{week_start}'
    )
    SELECT 
        GreyhoundID,
        SUM(CASE WHEN RaceNum <= 5 AND IsWinner = 1 THEN 1 ELSE 0 END) as RawWins,
        COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
    FROM dog_form_raw
    GROUP BY GreyhoundID
    """
    
    form_df = pd.read_sql_query(form_query, conn)
    
    # Get week's races
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
    WHERE ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '{week_start}'
      AND rm.MeetingDate <= '{week_end}'
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
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
    df = df[(df['StartingPrice'] >= PRICE_MIN) & (df['StartingPrice'] <= PRICE_MAX)]
    
    if len(df) == 0:
        continue
    
    # Calculate scores (NO track scaling - just raw performance)
    # Normalize pace within this week's candidates
    pace_min = df['RawPaceAvg'].min()
    pace_max = df['RawPaceAvg'].max()
    
    if pace_max - pace_min > 0:
        df['PaceScore'] = (df['RawPaceAvg'] - pace_min) / (pace_max - pace_min)
    else:
        df['PaceScore'] = 0.5
    
    # Form rate (wins / races)
    df['FormRate'] = df.apply(
        lambda row: min((row['RawWins'] / row['FormRaces'] * 100), 100) if pd.notna(row['FormRaces']) and row['FormRaces'] > 0 else 0,
        axis=1
    )
    df['FormScore'] = df['FormRate'] / 100.0
    
    # Weighted score
    df['WeightedScore'] = (df['PaceScore'] * PACE_WEIGHT) + (df['FormScore'] * FORM_WEIGHT)
    
    # Filter to threshold
    high_conf = df[df['WeightedScore'] >= THRESHOLD].copy()
    
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

# Calculate days in period
days_in_period = (datetime(2025, 12, 1) - datetime(2025, 6, 1)).days

progress(f"\n{'='*100}")
progress(f"RESULTS: Threshold {THRESHOLD} | ${PRICE_MIN}-${PRICE_MAX}")
progress(f"{'='*100}")

total_bets = len(results)
total_wins = results['ActualWinner'].sum()
strike_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
avg_odds = results['StartingPrice'].mean()
total_profit = results['Profit'].sum()
roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
daily_avg = total_bets / days_in_period

progress(f"\nTotal Bets: {total_bets} ({daily_avg:.1f}/day)", indent=1)
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
    
    progress(f"{month}: {len(month_data):3} bets | {wins:2} wins | {strike:5.1f}% | ${avg_odds_month:.2f} | {profit:+6.1f} | ROI: {roi_month:+.1f}%", indent=1)

# Price bracket breakdown
progress(f"\n{'='*100}")
progress("BREAKDOWN BY PRICE BRACKET")
progress(f"{'='*100}")

price_brackets = [
    (2.50, 3.00, "$2.50-$3.00"),
    (3.00, 3.50, "$3.00-$3.50"),
    (3.50, 4.00, "$3.50-$4.00"),
    (4.00, 4.50, "$4.00-$4.50"),
    (4.50, 5.00, "$4.50-$5.00"),
]

for min_p, max_p, label in price_brackets:
    bracket = results[(results['StartingPrice'] >= min_p) & (results['StartingPrice'] < max_p)]
    if len(bracket) > 0:
        wins = bracket['ActualWinner'].sum()
        strike = wins / len(bracket) * 100
        avg_odds_b = bracket['StartingPrice'].mean()
        profit = bracket['Profit'].sum()
        roi_b = profit / len(bracket) * 100
        progress(f"{label}: {len(bracket):3} bets | {wins:2} wins | {strike:5.1f}% | {profit:+6.1f} | ROI: {roi_b:+.1f}%", indent=1)

# Export
csv_file = 'backtest_optimized_bets.csv'
export_cols = ['RaceDate', 'GreyhoundName', 'TrackName', 'StartingPrice', 
               'WeightedScore', 'PaceScore', 'FormScore', 'ActualWinner', 'Profit']
results[export_cols].to_csv(csv_file, index=False)
progress(f"\nExported {len(results)} bets to {csv_file}")
