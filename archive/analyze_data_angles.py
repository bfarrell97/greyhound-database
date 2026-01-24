"""
Analyze database for unexplored betting angles
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('greyhound_racing.db')

print("="*80)
print("DATABASE SCHEMA & DATA AVAILABILITY ANALYSIS")
print("="*80)

# Get schema
cursor = conn.cursor()

# GreyhoundEntries columns
cursor.execute('PRAGMA table_info(GreyhoundEntries)')
cols = [r[1] for r in cursor.fetchall()]
print(f'\nGreyhoundEntries columns:\n  {cols}\n')

# Races columns  
cursor.execute('PRAGMA table_info(Races)')
cols = [r[1] for r in cursor.fetchall()]
print(f'Races columns:\n  {cols}\n')

# RaceMeetings columns
cursor.execute('PRAGMA table_info(RaceMeetings)')
cols = [r[1] for r in cursor.fetchall()]
print(f'RaceMeetings columns:\n  {cols}\n')

# Tracks columns
cursor.execute('PRAGMA table_info(Tracks)')
cols = [r[1] for r in cursor.fetchall()]
print(f'Tracks columns:\n  {cols}\n')

# Trainers columns
cursor.execute('PRAGMA table_info(Trainers)')
cols = [r[1] for r in cursor.fetchall()]
print(f'Trainers columns:\n  {cols}\n')

# Data availability analysis
print("="*80)
print("DATA AVAILABILITY (sampling 100k entries)")
print("="*80)

df = pd.read_sql_query("""
    SELECT 
        ge.Box, ge.Weight, ge.Position, ge.FinishTime, ge.Split, ge.InRun,
        ge.StartingPrice, ge.FinishTimeBenchmarkLengths, ge.SplitBenchmarkLengths,
        ge.EarlySpeed, ge.Rating, ge.Margin,
        r.Distance, r.Grade,
        t.TrackName,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    ORDER BY RANDOM()
    LIMIT 100000
""", conn)

print(f"\nSample size: {len(df):,}\n")
for col in df.columns:
    non_null = df[col].notna().sum()
    non_empty = (df[col].notna() & (df[col] != '')).sum()
    pct = 100 * non_empty / len(df)
    print(f"  {col}: {non_empty:,} ({pct:.1f}%)")

# Check what distances exist
print("\n" + "="*80)
print("DISTANCE BREAKDOWN")
print("="*80)
dist_counts = df['Distance'].value_counts().head(15)
print(dist_counts)

# Check grades
print("\n" + "="*80)
print("GRADE BREAKDOWN (top 20)")
print("="*80)
grade_counts = df['Grade'].value_counts().head(20)
print(grade_counts)

# Check tracks
print("\n" + "="*80)
print("TRACK BREAKDOWN (top 20)")
print("="*80)
track_counts = df['TrackName'].value_counts().head(20)
print(track_counts)

# Box analysis
print("\n" + "="*80)
print("BOX WIN RATES (full dataset)")
print("="*80)

box_stats = pd.read_sql_query("""
    SELECT 
        Box,
        COUNT(*) as runs,
        SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN StartingPrice != '' AND StartingPrice IS NOT NULL 
            THEN CAST(StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries
    WHERE Box IS NOT NULL AND Box BETWEEN 1 AND 8
    GROUP BY Box
    ORDER BY Box
""", conn)
print(box_stats.to_string(index=False))

# Split time analysis - early speed advantage
print("\n" + "="*80)
print("SPLIT BENCHMARK vs WIN RATE")
print("="*80)

split_analysis = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN SplitBenchmarkLengths < -2 THEN 'Fast (<-2)'
            WHEN SplitBenchmarkLengths < -1 THEN 'Good (-2 to -1)'
            WHEN SplitBenchmarkLengths < 0 THEN 'Avg (-1 to 0)'
            WHEN SplitBenchmarkLengths < 1 THEN 'Slow (0 to 1)'
            ELSE 'Very Slow (>1)'
        END as split_category,
        COUNT(*) as runs,
        SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct
    FROM GreyhoundEntries
    WHERE SplitBenchmarkLengths IS NOT NULL
    GROUP BY split_category
    ORDER BY MIN(SplitBenchmarkLengths)
""", conn)
print(split_analysis.to_string(index=False))

# Trainer analysis
print("\n" + "="*80)
print("TOP TRAINERS BY WIN RATE (min 500 runs)")
print("="*80)

trainer_stats = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    GROUP BY tr.TrainerName
    HAVING COUNT(*) >= 500
    ORDER BY win_pct DESC
    LIMIT 20
""", conn)
print(trainer_stats.to_string(index=False))

# Track-specific performance
print("\n" + "="*80)
print("TRACK WIN RATES")
print("="*80)

track_stats = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 1000
    ORDER BY runs DESC
    LIMIT 25
""", conn)
print(track_stats.to_string(index=False))

# Favorite analysis - does SP predict winner?
print("\n" + "="*80)
print("STARTING PRICE BANDS vs WIN RATE")
print("="*80)

sp_analysis = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN CAST(StartingPrice AS REAL) < 2 THEN '$1.01-$1.99'
            WHEN CAST(StartingPrice AS REAL) < 3 THEN '$2.00-$2.99'
            WHEN CAST(StartingPrice AS REAL) < 4 THEN '$3.00-$3.99'
            WHEN CAST(StartingPrice AS REAL) < 5 THEN '$4.00-$4.99'
            WHEN CAST(StartingPrice AS REAL) < 7 THEN '$5.00-$6.99'
            WHEN CAST(StartingPrice AS REAL) < 10 THEN '$7.00-$9.99'
            WHEN CAST(StartingPrice AS REAL) < 15 THEN '$10.00-$14.99'
            WHEN CAST(StartingPrice AS REAL) < 25 THEN '$15.00-$24.99'
            ELSE '$25.00+'
        END as sp_band,
        COUNT(*) as runs,
        SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        ROUND(100.0 / AVG(CAST(StartingPrice AS REAL)), 2) as implied_prob,
        ROUND(100.0 * SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) / COUNT(*) - 
              100.0 / AVG(CAST(StartingPrice AS REAL)), 2) as edge
    FROM GreyhoundEntries
    WHERE StartingPrice != '' AND StartingPrice IS NOT NULL
      AND CAST(StartingPrice AS REAL) > 1
    GROUP BY sp_band
    ORDER BY MIN(CAST(StartingPrice AS REAL))
""", conn)
print(sp_analysis.to_string(index=False))

# Days since last run
print("\n" + "="*80)
print("UNEXPLORED ANGLES TO INVESTIGATE")
print("="*80)
print("""
1. BOX DRAW BIAS BY TRACK
   - Some tracks favor inside boxes (rail advantage)
   - Some favor wide boxes (cleaner runs)
   
2. EARLY SPEED (SPLIT BENCHMARK)
   - Dogs with fast early speed (SplitBenchmarkLengths < -1) may have edge
   - Especially at short distances (300m, 350m)
   
3. TRAINER PATTERNS
   - Some trainers win at specific tracks
   - Trainer + Track combinations
   
4. DISTANCE SPECIALISTS  
   - Dogs performing better at specific distances
   - Step up/down in distance analysis
   
5. WEIGHT CHANGES
   - Weight gain/loss between races
   - Optimal racing weight by dog
   
6. GRADE CLASS
   - Class drop = advantage?
   - First time at grade
   
7. TRACK-SPECIFIC FORM
   - Performance at specific track vs overall form
   - Track experience (# of runs at track)
   
8. RACE NUMBER (TIME OF DAY)
   - Early races vs late races
   - Fresh dogs vs tired track
""")

conn.close()
print("\nAnalysis complete!")
