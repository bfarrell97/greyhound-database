"""
Check for historical early speed features in the database
Look for firstsec_GM_ADJ or similar metrics
"""

import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

print("="*100)
print("CHECKING FOR EARLY SPEED FEATURES")
print("="*100)

# Check if Greyhounds table has any early speed metrics
print("\n1. Greyhounds table columns:")
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(Greyhounds)')
cols = [(c[1], c[2]) for c in cursor.fetchall()]
for col, type_ in cols:
    print(f"  {col:30} {type_}")

# Check what we can calculate from historical race data
print("\n2. What we can calculate from GreyhoundEntries:")

# For each greyhound, calculate average first sectional (Split) from last races
query = """
SELECT 
    g.GreyhoundID,
    g.GreyhoundName,
    COUNT(*) as total_races,
    COUNT(ge.Split) as races_with_split,
    AVG(ge.Split) as avg_split,
    ROUND(AVG(ge.Split), 2) as avg_split_rounded,
    MIN(ge.Split) as min_split,
    MAX(ge.Split) as max_split
FROM Greyhounds g
LEFT JOIN GreyhoundEntries ge ON g.GreyhoundID = ge.GreyhoundID
GROUP BY g.GreyhoundID
HAVING COUNT(ge.Split) > 0
ORDER BY COUNT(ge.Split) DESC
LIMIT 20
"""

df = pd.read_sql_query(query, conn)
print("\nGreyhounds with historical split data:")
print(df)

# Check if we can get last 5 races split for a specific dog
print("\n3. Example: Last 5 races split for first greyhound:")

query_last5 = """
SELECT 
    g.GreyhoundID,
    g.GreyhoundName,
    ge.EntryID,
    ge.Split,
    ge.FinishTime,
    ge.Position,
    r.Distance,
    t.TrackName,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE g.GreyhoundID = (
    SELECT g2.GreyhoundID FROM Greyhounds g2
    LEFT JOIN GreyhoundEntries ge2 ON g2.GreyhoundID = ge2.GreyhoundID
    GROUP BY g2.GreyhoundID
    HAVING COUNT(ge2.Split) > 5
    ORDER BY COUNT(ge2.Split) DESC
    LIMIT 1
)
  AND ge.Split IS NOT NULL
ORDER BY rm.MeetingDate DESC
LIMIT 5
"""

df_last5 = pd.read_sql_query(query_last5, conn)
print(df_last5[['GreyhoundName', 'MeetingDate', 'Split', 'TrackName', 'Distance', 'Position']])

# Check if we can correlate historical average split with win rate
print("\n4. Correlation: Average Historical Split vs Win Rate:")

query_correlation = """
WITH dog_stats AS (
    SELECT 
        g.GreyhoundID,
        g.GreyhoundName,
        AVG(CASE WHEN ge.Position = '1' THEN 1.0 ELSE 0.0 END) as win_rate,
        AVG(ge.Split) as avg_split,
        COUNT(*) as races
    FROM Greyhounds g
    LEFT JOIN GreyhoundEntries ge ON g.GreyhoundID = ge.GreyhoundID
    WHERE ge.Split IS NOT NULL
    GROUP BY g.GreyhoundID
    HAVING COUNT(*) >= 10
)
SELECT 
    GreyhoundName,
    win_rate,
    avg_split,
    races
FROM dog_stats
ORDER BY avg_split DESC
LIMIT 20
"""

df_corr = pd.read_sql_query(query_correlation, conn)
print(df_corr)

# Calculate actual correlation coefficient
if len(df_corr) > 0 and df_corr['win_rate'].notna().sum() > 0:
    correlation = df_corr['win_rate'].corr(df_corr['avg_split'])
    print(f"\nPearson correlation (avg_split vs win_rate): {correlation:.3f}")
    print("Note: Positive correlation = dogs with faster historical splits win more often")

conn.close()

print("\n" + "="*100)
print("NEXT STEP")
print("="*100)
print("""
We can build a feature: "LastN_Avg_Split"
- For each dog, calculate average Split from last 3-5 races
- Use this in ML model to predict whether a dog will have good early speed
- This is PREDICTIVE because it's based on historical form, not current race data
- Similar to how we calculate LastN_WinRate, LastN_AvgPosition, etc.
""")
