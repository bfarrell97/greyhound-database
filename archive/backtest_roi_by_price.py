"""
BACKTEST: Weighted Model by Price Category
Shows ROI across different starting price ranges
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*120)
progress("BACKTEST: Weighted Model (70% Pace + 30% Form) - ROI by Price Category")
progress("="*120)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all historical races with pace and form metrics
query = """
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
),

dog_pace_avg AS (
    SELECT 
        GreyhoundID,
        GreyhoundName,
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
        ge.StartingPrice,
        rm.MeetingDate,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '2025-01-01'
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
)

SELECT 
    dpa.GreyhoundName,
    dpa.HistoricalPaceAvg,
    COALESCE(dfl.RecentWins, 0) as RecentWins,
    COALESCE(dfl.RecentRaces, 0) as RecentRaces,
    COALESCE(dfl.RecentWinRate, 0) as RecentWinRate,
    drf.IsWinner,
    drf.StartingPrice,
    drf.MeetingDate,
    ROW_NUMBER() OVER (PARTITION BY dpa.GreyhoundID ORDER BY drf.MeetingDate DESC) as RaceNum
FROM dog_pace_avg dpa
JOIN dog_recent_form drf ON dpa.GreyhoundID = drf.GreyhoundID
LEFT JOIN dog_form_last_5 dfl ON dpa.GreyhoundID = dfl.GreyhoundID
WHERE drf.RaceNum > 1
  AND drf.MeetingDate >= '2025-01-01'
"""

progress("\nLoading 2025 race data...", indent=1)
df = pd.read_sql_query(query, conn)
conn.close()

# Clean data
df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
df['RecentWinRate'] = pd.to_numeric(df['RecentWinRate'], errors='coerce')

df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])
progress(f"Loaded {len(df):,} races", indent=1)

# Calculate normalized scores
pace_min = df['HistoricalPaceAvg'].min()
pace_max = df['HistoricalPaceAvg'].max()
df['PaceScore'] = (df['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)
df['FormScore'] = df['RecentWinRate'] / 100.0

# Weighted score
df['WeightedScore'] = (df['PaceScore'] * 0.7) + (df['FormScore'] * 0.3)

# Define price categories
price_categories = [
    (1.01, 1.50, "$1.01-$1.50"),
    (1.51, 2.00, "$1.51-$2.00"),
    (2.01, 3.00, "$2.01-$3.00"),
    (3.01, 4.00, "$3.01-$4.00"),
    (4.01, 5.00, "$4.01-$5.00"),
    (5.01, 10.00, "$5.01-$10.00"),
    (10.01, 100.00, "$10.01+"),
]

progress("\n" + "="*120)
progress("ROI BY PRICE CATEGORY (Weighted Score >= 0.6)")
progress("="*120 + "\n")

results = []

for min_price, max_price, label in price_categories:
    subset = df[
        (df['WeightedScore'] >= 0.6) &
        (df['StartingPrice'] >= min_price) &
        (df['StartingPrice'] <= max_price)
    ].copy()
    
    if len(subset) == 0:
        continue
    
    wins = subset['IsWinner'].sum()
    bets = len(subset)
    strike_rate = wins / bets * 100
    avg_odds = subset['StartingPrice'].mean()
    
    # Calculate ROI
    total_return = (wins * avg_odds) + (bets - wins) * 0.0
    roi = ((total_return - bets) / bets) * 100
    
    # Expected value
    win_rate = wins / bets
    ev = (win_rate * avg_odds) - 1
    
    results.append({
        'Category': label,
        'Bets': bets,
        'Wins': wins,
        'Strike%': strike_rate,
        'AvgOdds': avg_odds,
        'ROI%': roi,
        'EV': ev,
    })
    
    status = "[EXCELLENT]" if roi > 30 else "[GOOD]" if roi > 10 else "[OK]" if roi > 0 else "[BAD]"
    print(f"{label:15} | Bets: {bets:5} | Wins: {wins:4} ({strike_rate:5.1f}%) | AvgOdds: ${avg_odds:5.2f} | ROI: {roi:+6.1f}% {status}")

# Summary stats
results_df = pd.DataFrame(results)

print("\n" + "="*120)
print("SUMMARY BY ROI PERFORMANCE")
print("="*120 + "\n")

# Sort by ROI
results_df_sorted = results_df.sort_values('ROI%', ascending=False)

print(results_df_sorted.to_string(index=False))

print("\n" + "="*120)
print("OVERALL STATISTICS")
print("="*120)

total_bets = results_df['Bets'].sum()
total_wins = results_df['Wins'].sum()
overall_strike = (total_wins / total_bets * 100)
weighted_avg_odds = (results_df['Bets'] * results_df['AvgOdds']).sum() / total_bets
overall_return = (total_wins * weighted_avg_odds) + (total_bets - total_wins) * 0.0
overall_roi = ((overall_return - total_bets) / total_bets) * 100

progress(f"Total Bets: {total_bets:,}", indent=1)
progress(f"Total Wins: {total_wins:,}", indent=1)
progress(f"Overall Strike Rate: {overall_strike:.1f}%", indent=1)
progress(f"Weighted Average Odds: ${weighted_avg_odds:.2f}", indent=1)
progress(f"Overall ROI: {overall_roi:+.1f}%", indent=1)

progress("\n" + "="*120)
progress("BEST PRICE RANGE")
progress("="*120)

best_row = results_df_sorted.iloc[0]
progress(f"Category: {best_row['Category']}", indent=1)
progress(f"ROI: {best_row['ROI%']:+.1f}%", indent=1)
progress(f"Strike Rate: {best_row['Strike%']:.1f}%", indent=1)
progress(f"Bets: {int(best_row['Bets'])}", indent=1)

progress("\n" + "="*120)
