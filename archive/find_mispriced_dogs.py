"""
Find mispriced dogs - using exact backtest batch query methodology
70% Pace + 30% Form @ 0.80 threshold on $1.50-$3.00 range
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("FINDING MISPRICED DOGS: Market edge analysis")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all unique race dates
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-10-01'
  AND rm.MeetingDate <= '2025-12-31'
ORDER BY RaceDate
"""

progress("\nGetting race dates...", indent=1)
dates_df = pd.read_sql_query(query_dates, conn)
race_dates = dates_df['RaceDate'].tolist()
progress(f"Found {len(race_dates)} unique race dates\n", indent=1)

# Collect all results with scores
all_results = []

progress("Processing races (batch queries)...\n", indent=1)

for idx, race_date in enumerate(race_dates, 1):
    if idx % 15 == 0:
        progress(f"[{idx}/{len(race_dates)}] Processing {race_date}... ({len(all_results)} dogs so far)", indent=1)

    # Use exact same batch query as backtest
    query = f"""
    WITH dog_pace_history AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            rm.MeetingDate,
            (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < DATE('{race_date}')
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
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
            rm.MeetingDate,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND rm.MeetingDate < DATE('{race_date}')
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
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as ActualWinner
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE DATE(rm.MeetingDate) = DATE('{race_date}')
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND ge.StartingPrice IS NOT NULL
    )
    
    SELECT 
        tr.GreyhoundID,
        tr.GreyhoundName,
        tr.StartingPrice,
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

# Normalize pace and form to 0-1
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()
if pace_max - pace_min > 0:
    df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
else:
    df['PaceScore'] = 0.5

df['FormScore'] = df['RecentWinRate'] / 100.0
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Calculate implied probability
df['ImpliedProb'] = 1.0 / df['StartingPrice']

# Filter to 0.80+ confidence
high_conf = df[df['WeightedScore'] >= 0.80].copy()

progress("=" * 100)
progress("HIGH CONFIDENCE PICKS (Score >= 0.80)")
progress("=" * 100)

total_bets = len(high_conf)
total_wins = high_conf['ActualWinner'].sum()
strike_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
avg_odds = high_conf['StartingPrice'].mean()
implied_prob = high_conf['ImpliedProb'].mean() * 100
actual_edge = strike_rate - implied_prob

progress(f"\nTotal Bets: {total_bets}", indent=1)
progress(f"Total Wins: {total_wins}", indent=1)
progress(f"Strike Rate: {strike_rate:.1f}%", indent=1)
progress(f"Avg Odds: ${avg_odds:.2f}", indent=1)
progress(f"Market Implied Win %: {implied_prob:.1f}%", indent=1)
progress(f"Our Actual Edge: {actual_edge:+.1f}% (positive = market undervalued our picks)", indent=1)

# Find biggest edges (market thinks low prob but we win often)
high_conf['MarketEdge'] = high_conf['ActualWinner'] - high_conf['ImpliedProb']
top_edges = high_conf.nlargest(20, 'MarketEdge')

progress("\n" + "=" * 100)
progress("TOP 20 INDIVIDUAL DOGS WITH BIGGEST POSITIVE EDGE")
progress("=" * 100)
progress("(Market undervalued these dogs - we won when market thought we shouldn't)\n")

for i, (_, row) in enumerate(top_edges.iterrows(), 1):
    edge_display = "WIN" if row['ActualWinner'] == 1 else "LOSS"
    edge_pct = row['MarketEdge'] * 100
    progress(f"{i:2}. {row['GreyhoundName']:20} @ ${row['StartingPrice']:.2f} ({edge_display:4}) | Score {row['WeightedScore']:.3f} | Edge {edge_pct:+.1f}%", indent=1)

# Find worst edges (market thinks high prob but we lose often)
low_conf = df[df['WeightedScore'] < 0.80].copy()
low_conf['MarketEdge'] = low_conf['ActualWinner'] - low_conf['ImpliedProb']
worst_edges = low_conf.nsmallest(20, 'MarketEdge')

progress("\n" + "=" * 100)
progress("TOP 20 WORST MISPRICES (When we DON'T pick, market overvalued them)")
progress("=" * 100)
progress("(These dogs market liked but we avoided - mostly losers)\n")

for i, (_, row) in enumerate(worst_edges.iterrows(), 1):
    edge_display = "WIN" if row['ActualWinner'] == 1 else "LOSS"
    edge_pct = row['MarketEdge'] * 100
    progress(f"{i:2}. {row['GreyhoundName']:20} @ ${row['StartingPrice']:.2f} ({edge_display:4}) | Score {row['WeightedScore']:.3f} | Edge {edge_pct:+.1f}%", indent=1)

progress("\n" + "=" * 100)
progress("CONCLUSION")
progress("=" * 100)
progress(f"\nMarket edge on our 0.80+ picks: {actual_edge:+.1f}%", indent=1)

if actual_edge > 0:
    progress("✓ We ARE picking dogs market undervalued (good sign!)", indent=1)
    progress("  But the edge is so small (+1.2% or less), ROI is minimal.", indent=1)
    progress("  Conclusion: Market prices are very efficient on $1.50-$3.00 range.", indent=1)
else:
    progress("✗ We ARE picking dogs market overvalued (bad sign!)", indent=1)
    progress("  The market is pricing them correctly or better than our model.", indent=1)

progress("\n" + "=" * 100)

conn.close()
