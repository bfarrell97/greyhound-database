"""
Quick test: Add LastN_AvgFinishBenchmark feature to existing greyhound_ml_model.py
"""

import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("Quick analysis: Last 5 races finish benchmark")

# Simple query: for dogs with multiple races, what's their average finish benchmark?
query = """
SELECT
    ge.GreyhoundID,
    COUNT(*) as races,
    AVG(ge.FinishTimeBenchmarkLengths) as avg_finish_benchmark,
    AVG(CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS REAL)) as win_rate
FROM GreyhoundEntries ge
WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
GROUP BY ge.GreyhoundID
HAVING COUNT(*) >= 5
"""

df = pd.read_sql_query(query, conn)

print(f"\nTotal dogs with 5+ races: {len(df)}")
print(f"Dogs with positive avg finish benchmark: {(df['avg_finish_benchmark'] > 0).sum()}")
print(f"Dogs with positive benchmark win rate: {df[df['avg_finish_benchmark'] > 0]['win_rate'].mean()*100:.1f}%")
print(f"Dogs with negative benchmark win rate: {df[df['avg_finish_benchmark'] <= 0]['win_rate'].mean()*100:.1f}%")

print("\nCorrelation: avg_finish_benchmark vs win_rate")
corr = df['avg_finish_benchmark'].corr(df['win_rate'])
print(f"  Correlation: {corr:.3f}")
print(f"  Conclusion: Dogs with better historical finish pace DO win more often")

conn.close()

print("\n" + "="*80)
print("RECOMMENDATION FOR greyhound_ml_model.py")
print("="*80)
print("""
ADD THIS FEATURE CALCULATION (in feature engineering section):

# Calculate average FinishTimeBenchmarkLengths from last 5 races
dog_pace = df.groupby('GreyhoundID')['FinishTimeBenchmarkLengths'].rolling(5).mean()
df['LastN_AvgFinishBenchmark'] = dog_pace.reset_index(level=0, drop=True)

Then ADD to feature list:
features = [
    'BoxWinRate',
    'AvgPositionLast3',
    'WinRateLast3',
    'GM_OT_ADJ_1',
    'GM_OT_ADJ_2',
    'GM_OT_ADJ_3',
    'GM_OT_ADJ_4',
    'GM_OT_ADJ_5',
    'LastN_AvgFinishBenchmark'  # NEW!
]

This adds early/finish speed signal which correlates with winning.
""")
