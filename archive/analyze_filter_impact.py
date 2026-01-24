"""
ANALYZE: Which dogs are affected by outlier filtering?
Find which high-quality dogs are losing pace metrics due to filtering
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get dogs with and without filtering
query_without = """
SELECT 
    dph.GreyhoundName,
    COUNT(*) as TotalRaces,
    COUNT(DISTINCT CASE WHEN dph.RaceNum <= 5 THEN dph.MeetingDate END) as PacesUsed,
    AVG(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as HistoricalPaceAvg,
    MAX(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as MaxPace,
    MIN(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as MinPace
FROM (
    SELECT 
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
) dph
GROUP BY dph.GreyhoundName
HAVING PacesUsed >= 5
ORDER BY HistoricalPaceAvg DESC
"""

query_with = """
SELECT 
    dph.GreyhoundName,
    COUNT(*) as TotalRaces,
    COUNT(DISTINCT CASE WHEN dph.RaceNum <= 5 THEN dph.MeetingDate END) as PacesUsed,
    AVG(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as HistoricalPaceAvg,
    MAX(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as MaxPace,
    MIN(CASE WHEN dph.RaceNum <= 5 THEN dph.TotalFinishBench END) as MinPace
FROM (
    SELECT 
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
      AND ge.FinishTimeBenchmarkLengths BETWEEN -20.52 AND 9.18
) dph
GROUP BY dph.GreyhoundName
HAVING PacesUsed >= 5
ORDER BY HistoricalPaceAvg DESC
"""

print("Loading dog pace data WITHOUT filter...")
df_without = pd.read_sql_query(query_without, conn)

print("Loading dog pace data WITH filter...")
df_with = pd.read_sql_query(query_with, conn)

# Merge to find differences
comparison = pd.merge(
    df_without.rename(columns={col: f'{col}_without' for col in df_without.columns if col != 'GreyhoundName'}),
    df_with.rename(columns={col: f'{col}_with' for col in df_with.columns if col != 'GreyhoundName'}),
    on='GreyhoundName',
    how='outer'
)

# Find dogs affected by filtering
comparison['PaceChange'] = comparison['HistoricalPaceAvg_with'] - comparison['HistoricalPaceAvg_without']
comparison['IsAffected'] = comparison['PaceChange'].abs() > 0.01

affected = comparison[comparison['IsAffected']].copy()
affected = affected.sort_values('PaceChange', ascending=True)

print("\n" + "="*120)
print("DOGS MOST NEGATIVELY AFFECTED BY OUTLIER FILTERING")
print("="*120)
print(f"\nTotal dogs affected: {len(affected)}")
print(f"Total dogs in database: {len(comparison)}")

print("\nTOP 20 DOGS LOSING MOST PACE:")
for idx, row in affected.head(20).iterrows():
    print(f"\n{row['GreyhoundName']:30s} | Pace Change: {row['PaceChange']:+7.3f}")
    print(f"  Without filter: {row['HistoricalPaceAvg_without']:+7.3f} (range {row['MinPace_without']:+7.3f} to {row['MaxPace_without']:+7.3f})")
    print(f"  With filter:   {row['HistoricalPaceAvg_with']:+7.3f} (range {row['MinPace_with']:+7.3f} to {row['MaxPace_with']:+7.3f})")
    print(f"  Races used: {row['PacesUsed_without']:.0f} -> {row['PacesUsed_with']:.0f}")

print("\n" + "="*120)
print("TOP 20 DOGS GAINING MOST PACE:")
for idx, row in affected.tail(20).sort_values('PaceChange', ascending=False).iterrows():
    print(f"\n{row['GreyhoundName']:30s} | Pace Change: {row['PaceChange']:+7.3f}")
    print(f"  Without filter: {row['HistoricalPaceAvg_without']:+7.3f} (range {row['MinPace_without']:+7.3f} to {row['MaxPace_without']:+7.3f})")
    print(f"  With filter:   {row['HistoricalPaceAvg_with']:+7.3f} (range {row['MinPace_with']:+7.3f} to {row['MaxPace_with']:+7.3f})")
    print(f"  Races used: {row['PacesUsed_without']:.0f} -> {row['PacesUsed_with']:.0f}")

# Summary statistics
print("\n" + "="*120)
print("OVERALL IMPACT SUMMARY")
print("="*120)
print(f"\nMean pace before filter: {comparison['HistoricalPaceAvg_without'].mean():+.3f}")
print(f"Mean pace after filter:  {comparison['HistoricalPaceAvg_with'].mean():+.3f}")
print(f"Mean change: {(comparison['HistoricalPaceAvg_with'].mean() - comparison['HistoricalPaceAvg_without'].mean()):+.3f}")

print(f"\nDogs losing pace (avg): {affected[affected['PaceChange'] < 0]['PaceChange'].mean():+.3f}")
print(f"Dogs gaining pace (avg): {affected[affected['PaceChange'] > 0]['PaceChange'].mean():+.3f}")

conn.close()
