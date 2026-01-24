"""
Check what actual data is available for early speed analysis
Focus on Sectional100m (early speed proxy) since it's 100m time
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

print("="*100)
print("SECTIONAL TIMES TABLE ANALYSIS")
print("="*100)

# Check SectionalTimes data availability
query = """
SELECT 
    COUNT(*) as total_rows,
    COUNT(Sectional100m) as sectional_100m_count,
    COUNT(Sectional200m) as sectional_200m_count,
    COUNT(Sectional300m) as sectional_300m_count,
    COUNT(Sectional400m) as sectional_400m_count,
    COUNT(FinishTime) as finish_time_count,
    COUNT(Split) as split_count
FROM SectionalTimes
"""

result = pd.read_sql_query(query, conn)
print("\nSectionalTimes data availability:")
print(result)

# Check if there's any sectional data at all
query2 = """
SELECT 
    MIN(Sectional100m) as min_100m,
    MAX(Sectional100m) as max_100m,
    AVG(Sectional100m) as avg_100m,
    MIN(Sectional200m) as min_200m,
    MAX(Sectional200m) as max_200m,
    AVG(Sectional200m) as avg_200m,
    MIN(Sectional300m) as min_300m,
    MAX(Sectional300m) as max_300m,
    AVG(Sectional300m) as avg_300m
FROM SectionalTimes
WHERE Sectional100m IS NOT NULL OR Sectional200m IS NOT NULL OR Sectional300m IS NOT NULL
"""

result2 = pd.read_sql_query(query2, conn)
print("\nSectional values (if any):")
print(result2)

print("\n" + "="*100)
print("GREYHOUND ENTRIES TABLE - WHAT'S AVAILABLE FOR EARLY SPEED")
print("="*100)

# GreyhoundEntries has EarlySpeed and Rating but they're all NULL
# But it has Split which is essentially the first sectional (early pace)
query3 = """
SELECT 
    COUNT(*) as total_entries,
    COUNT(Split) as split_count,
    COUNT(DISTINCT Split) as distinct_split_values,
    MIN(Split) as min_split,
    MAX(Split) as max_split,
    AVG(Split) as avg_split,
    COUNT(FinishTime) as finish_time_count,
    COUNT(DISTINCT FinishTime) as distinct_finish_times
FROM GreyhoundEntries
"""

result3 = pd.read_sql_query(query3, conn)
print("\nGreyhoundEntries early speed proxies:")
print(result3)

# Get some sample data
print("\nSample GreyhoundEntries with Split data:")
query4 = """
SELECT 
    EntryID, Box, Weight, FinishTime, Split, Position
FROM GreyhoundEntries 
WHERE Split IS NOT NULL
LIMIT 10
"""

result4 = pd.read_sql_query(query4, conn)
print(result4)

print("\n" + "="*100)
print("BENCHMARKS TABLE - SPLIT/TIME BENCHMARKS")
print("="*100)

query5 = """
SELECT 
    COUNT(*) as total_benchmarks,
    COUNT(DISTINCT TrackName) as distinct_tracks,
    COUNT(DISTINCT Distance) as distinct_distances,
    AVG(AvgSplit) as avg_split_benchmark,
    AVG(AvgTime) as avg_finish_time_benchmark
FROM Benchmarks
"""

result5 = pd.read_sql_query(query5, conn)
print("\nBenchmarks overview:")
print(result5)

print("\nSample benchmarks (first sectional splits):")
query6 = """
SELECT 
    TrackName, Distance, AvgSplit, MedianSplit, FastestSplit, SampleSize
FROM Benchmarks
ORDER BY Distance
LIMIT 10
"""

result6 = pd.read_sql_query(query6, conn)
print(result6)

conn.close()

print("\n" + "="*100)
print("CONCLUSION: AVAILABLE EARLY SPEED METRICS")
print("="*100)
print("""
✓ AVAILABLE: GreyhoundEntries.Split (1,264,785 rows)
  - Represents the first sectional time (typically first split, which is early speed indicator)
  - Can be compared to Benchmarks.AvgSplit and Benchmarks.MedianSplit for track/distance
  
✓ AVAILABLE: Benchmarks table
  - Has AvgSplit, MedianSplit, FastestSplit by TrackName and Distance
  - Can calculate how a dog's Split compares to benchmark (early speed relative to field)
  
✗ NOT AVAILABLE: SectionalTimes table
  - Table structure exists but has 0 rows
  - So Sectional100m, Sectional200m, etc. don't have data
  
✗ NOT AVAILABLE: EarlySpeed and Rating columns
  - Exist in GreyhoundEntries but all NULL (0 non-null values)
  
RECOMMENDATION:
- Use GreyhoundEntries.Split as early speed proxy
- Calculate Split relative to Benchmark for that track/distance
- Higher split (relative to field average) = better early speed
""")
