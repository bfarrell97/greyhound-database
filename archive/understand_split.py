"""
Understand the Split metric better:
1. What does Split actually represent? (time to first marker)
2. How does it relate to winning in the CURRENT race?
3. Can we predict current race Split from historical data?
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*100)
print("UNDERSTANDING SPLIT AND SPLITBENCHMARKLENGTHS")
print("="*100)

# Check the relationship between Split and winning
print("\n1. Current Race: Split vs Position (lower split = better early pace?)")

query = """
SELECT
    ge.Position,
    COUNT(*) as count,
    AVG(ge.Split) as avg_split,
    MIN(ge.Split) as min_split,
    MAX(ge.Split) as max_split
FROM GreyhoundEntries ge
WHERE ge.Split IS NOT NULL
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
GROUP BY ge.Position
ORDER BY CAST(ge.Position AS INTEGER)
LIMIT 8
"""

df = pd.read_sql_query(query, conn)
print(df)

print("\n2. Relationship: SplitBenchmarkLengths vs Winning")
print("   (Positive SBL = faster than track average = better early speed)")

query2 = """
SELECT
    CASE WHEN ge.Position = '1' THEN 'Winner' ELSE 'Non-Winner' END as Result,
    COUNT(*) as count,
    AVG(ge.SplitBenchmarkLengths) as avg_split_benchmark,
    MIN(ge.SplitBenchmarkLengths) as min_split_benchmark,
    MAX(ge.SplitBenchmarkLengths) as max_split_benchmark
FROM GreyhoundEntries ge
WHERE ge.SplitBenchmarkLengths IS NOT NULL
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
GROUP BY CASE WHEN ge.Position = '1' THEN 'Winner' ELSE 'Non-Winner' END
"""

df2 = pd.read_sql_query(query2, conn)
print(df2)

# So positive Split benchmark = good. Now check if we can predict it from historical data
print("\n3. Can we predict current race SplitBenchmarkLengths from historical split?")

query3 = """
WITH dog_history AS (
    SELECT
        ge.GreyhoundID,
        ge.RaceID,
        rm.MeetingDate,
        ge.SplitBenchmarkLengths,
        -- Get average split from previous races
        AVG(LAG(ge.Split) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate)) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as prev_5_avg_split,
        -- Get win rate from previous races
        AVG(CAST((CASE WHEN LAG(ge.Position) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate) = '1' THEN 1 ELSE 0 END) AS REAL)) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) as prev_5_win_rate
    FROM GreyhoundEntries ge
    JOIN RaceMeetings rm ON ge.RaceID LIKE (SELECT RaceID FROM Races WHERE MeetingID = rm.MeetingID)
    WHERE ge.SplitBenchmarkLengths IS NOT NULL
    LIMIT 1000
)
SELECT 
    CORR(SplitBenchmarkLengths, prev_5_avg_split) as corr_split,
    CORR(SplitBenchmarkLengths, prev_5_win_rate) as corr_win_rate
FROM dog_history
WHERE prev_5_avg_split IS NOT NULL
"""

try:
    df3 = pd.read_sql_query(query3, conn)
    print(df3)
except Exception as e:
    print(f"Query error: {e}")
    print("\nLet's try a simpler approach...")

# Simple approach: for dogs with multiple races, check if historical split predicts current SBL
print("\n4. Simple check: Does historical average Split correlate with current SplitBenchmarkLengths?")

# Get recent races with SplitBenchmarkLengths
query4 = """
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.SplitBenchmarkLengths,
    ge.Split,
    ge.Position,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.SplitBenchmarkLengths IS NOT NULL
  AND ge.Split IS NOT NULL
  AND rm.MeetingDate >= '2025-11-01'
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""

df4 = pd.read_sql_query(query4, conn)

if len(df4) > 100:
    # Calculate correlation for each dog with 3+ races
    dog_corrs = []
    for dog_id in df4['GreyhoundID'].unique():
        dog_data = df4[df4['GreyhoundID'] == dog_id].sort_values('MeetingDate')
        if len(dog_data) >= 3:
            corr = dog_data['Split'].corr(dog_data['SplitBenchmarkLengths'])
            if pd.notna(corr):
                dog_corrs.append({'GreyhoundID': dog_id, 'correlation': corr, 'races': len(dog_data)})
    
    corr_df = pd.DataFrame(dog_corrs)
    print(f"Average correlation (Split vs SplitBenchmarkLengths): {corr_df['correlation'].mean():.3f}")
    print(f"Dogs analyzed: {len(corr_df)}")
    print(f"\nTop 10 by correlation:")
    print(corr_df.nlargest(10, 'correlation'))

conn.close()

print("\n" + "="*100)
print("KEY INSIGHT")
print("="*100)
print("""
Split represents the dog's actual first sectional time in that race.
SplitBenchmarkLengths shows if it was faster or slower than benchmark.

The problem: Split varies by:
- Track condition (wet/dry)
- Race distance
- Opposition quality
- Dog's current form

Better approach:
Instead of using historical Split directly, we should:
1. Use FinishTime (finish performance) to predict winning
2. Use SplitBenchmarkLengths as a FILTER (for races where we have it)
3. For upcoming races (where we don't have SBL yet), use other form features

The SplitBenchmarkLengths insight is valuable for FILTERING existing races,
but for predicting NEW races, we need other features that are available ahead of time.
""")
