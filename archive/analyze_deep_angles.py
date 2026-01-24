"""
Deep dive into unexplored betting angles
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('greyhound_racing.db')

print("="*80)
print("DEEP DIVE: UNEXPLORED BETTING ANGLES")
print("="*80)

# 1. BOX DRAW BY TRACK - Find tracks with extreme box bias
print("\n" + "="*80)
print("1. BOX BIAS BY TRACK (Box 1 vs Box 8 win rate differential)")
print("="*80)

box_by_track = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as total_runs,
        SUM(CASE WHEN ge.Box = 1 AND ge.Position = '1' THEN 1.0 ELSE 0 END) / 
            NULLIF(SUM(CASE WHEN ge.Box = 1 THEN 1.0 ELSE 0 END), 0) * 100 as box1_win_pct,
        SUM(CASE WHEN ge.Box = 8 AND ge.Position = '1' THEN 1.0 ELSE 0 END) / 
            NULLIF(SUM(CASE WHEN ge.Box = 8 THEN 1.0 ELSE 0 END), 0) * 100 as box8_win_pct
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Box IN (1, 8)
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 5000
    ORDER BY total_runs DESC
""", conn)

box_by_track['box1_advantage'] = box_by_track['box1_win_pct'] - box_by_track['box8_win_pct']
box_by_track = box_by_track.sort_values('box1_advantage', ascending=False)
print("\nStrongest Box 1 advantage tracks:")
print(box_by_track[['TrackName', 'total_runs', 'box1_win_pct', 'box8_win_pct', 'box1_advantage']].head(10).to_string(index=False))
print("\nStrongest Box 8 advantage tracks:")
print(box_by_track[['TrackName', 'total_runs', 'box1_win_pct', 'box8_win_pct', 'box1_advantage']].tail(5).to_string(index=False))

# 2. TRAINER + TRACK COMBINATIONS
print("\n" + "="*80)
print("2. TRAINER + TRACK SPECIALIZATION (min 100 runs at track)")
print("="*80)

trainer_track = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        t.TrackName,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    GROUP BY tr.TrainerName, t.TrackName
    HAVING COUNT(*) >= 100
    ORDER BY win_pct DESC
    LIMIT 30
""", conn)
print(trainer_track.to_string(index=False))

# 3. DISTANCE SPECIALIST - Dogs that excel at specific distances
print("\n" + "="*80)
print("3. DISTANCE PERFORMANCE PATTERNS")
print("="*80)

distance_win = pd.read_sql_query("""
    SELECT 
        r.Distance,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp,
        AVG(ge.FinishTimeBenchmarkLengths) as avg_ftb
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.Position != 'DNF'
    GROUP BY r.Distance
    HAVING COUNT(*) >= 10000
    ORDER BY runs DESC
""", conn)
print(distance_win.to_string(index=False))

# 4. RACE NUMBER ANALYSIS
print("\n" + "="*80)
print("4. RACE NUMBER (Position on card) - Favorite performance")
print("="*80)

race_num_analysis = pd.read_sql_query("""
    SELECT 
        r.RaceNumber,
        COUNT(*) as total_runs,
        SUM(CASE WHEN ge.Position = '1' AND CAST(ge.StartingPrice AS REAL) < 3 THEN 1 ELSE 0 END) as fav_wins,
        SUM(CASE WHEN CAST(ge.StartingPrice AS REAL) < 3 THEN 1 ELSE 0 END) as fav_runs,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' AND CAST(ge.StartingPrice AS REAL) < 3 THEN 1 ELSE 0 END) / 
              NULLIF(SUM(CASE WHEN CAST(ge.StartingPrice AS REAL) < 3 THEN 1 ELSE 0 END), 0), 2) as fav_win_pct
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' AND r.RaceNumber <= 12
    GROUP BY r.RaceNumber
    ORDER BY r.RaceNumber
""", conn)
print(race_num_analysis.to_string(index=False))

# 5. EARLY SPEED AT SHORT DISTANCES
print("\n" + "="*80)
print("5. EARLY SPEED (Split Benchmark) AT SHORT DISTANCES (<400m)")
print("="*80)

early_speed = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN SplitBenchmarkLengths < -2 THEN 'A: Very Fast (<-2)'
            WHEN SplitBenchmarkLengths < -1 THEN 'B: Fast (-2 to -1)'
            WHEN SplitBenchmarkLengths < 0 THEN 'C: Avg (-1 to 0)'
            WHEN SplitBenchmarkLengths < 1 THEN 'D: Slow (0 to 1)'
            ELSE 'E: Very Slow (>1)'
        END as split_category,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.SplitBenchmarkLengths IS NOT NULL
      AND r.Distance < 400
    GROUP BY split_category
    ORDER BY split_category
""", conn)
print(early_speed.to_string(index=False))

# 6. WEIGHT PATTERNS
print("\n" + "="*80)
print("6. WEIGHT DISTRIBUTION AND WIN RATES")
print("="*80)

weight_analysis = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN Weight < 26 THEN 'A: Light (<26kg)'
            WHEN Weight < 28 THEN 'B: Light-Med (26-28kg)'
            WHEN Weight < 30 THEN 'C: Medium (28-30kg)'
            WHEN Weight < 32 THEN 'D: Med-Heavy (30-32kg)'
            WHEN Weight < 34 THEN 'E: Heavy (32-34kg)'
            ELSE 'F: Very Heavy (34+kg)'
        END as weight_band,
        COUNT(*) as runs,
        SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN StartingPrice != '' THEN CAST(StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries
    WHERE Weight IS NOT NULL AND Weight > 20 AND Weight < 45
    GROUP BY weight_band
    ORDER BY weight_band
""", conn)
print(weight_analysis.to_string(index=False))

# 7. GRADE CLASS ANALYSIS
print("\n" + "="*80)
print("7. GRADE-SPECIFIC WIN RATES")
print("="*80)

grade_analysis = pd.read_sql_query("""
    SELECT 
        r.Grade,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) < 50 
            THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE r.Grade IS NOT NULL AND r.Grade != ''
    GROUP BY r.Grade
    HAVING COUNT(*) >= 5000
    ORDER BY runs DESC
""", conn)
print(grade_analysis.to_string(index=False))

# 8. TRACK EXPERIENCE - Does running at same track repeatedly help?
print("\n" + "="*80)
print("8. TRACK EXPERIENCE (# of prior runs at track)")
print("="*80)

# This is a more complex query - approximate with simpler analysis
track_exp = pd.read_sql_query("""
    WITH dog_track_runs AS (
        SELECT 
            ge.GreyhoundID,
            t.TrackID,
            COUNT(*) as track_runs
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        GROUP BY ge.GreyhoundID, t.TrackID
    )
    SELECT 
        CASE 
            WHEN track_runs = 1 THEN 'First time at track'
            WHEN track_runs <= 3 THEN '2-3 runs at track'
            WHEN track_runs <= 5 THEN '4-5 runs at track'
            WHEN track_runs <= 10 THEN '6-10 runs at track'
            ELSE '10+ runs at track'
        END as experience,
        COUNT(*) as dogs,
        AVG(track_runs) as avg_runs
    FROM dog_track_runs
    GROUP BY experience
    ORDER BY AVG(track_runs)
""", conn)
print(track_exp.to_string(index=False))

# 9. FIND VALUE: Dogs beating their odds
print("\n" + "="*80)
print("9. FINDING VALUE: Where does market misprice?")
print("="*80)

# Calculate ROI by various factors
roi_analysis = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN ge.Box = 1 THEN 'Box 1'
            WHEN ge.Box = 8 THEN 'Box 8'
            ELSE 'Box 2-7'
        END as box_group,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    WHERE ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 5
    GROUP BY box_group
    ORDER BY box_group
""", conn)
print("\nROI by Box (odds $1.50-$5.00):")
print(roi_analysis.to_string(index=False))

# ROI by track
roi_track = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 5
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 1000
    ORDER BY roi DESC
    LIMIT 15
""", conn)
print("\nBest ROI by Track (odds $1.50-$5.00, min 1000 bets):")
print(roi_track.to_string(index=False))

# 10. COMBINATION ANALYSIS - Multiple factors
print("\n" + "="*80)
print("10. MULTI-FACTOR COMBINATIONS")
print("="*80)

combo_analysis = pd.read_sql_query("""
    SELECT 
        CASE WHEN ge.Box = 1 THEN 'Box1' ELSE 'Box2-8' END as box_type,
        CASE WHEN CAST(ge.StartingPrice AS REAL) < 3 THEN 'Fav<$3' ELSE 'Other' END as price_type,
        CASE WHEN r.Distance < 400 THEN 'Sprint' ELSE 'Middle/Stay' END as dist_type,
        COUNT(*) as runs,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) > 1
    GROUP BY box_type, price_type, dist_type
    HAVING COUNT(*) >= 10000
    ORDER BY roi DESC
""", conn)
print(combo_analysis.to_string(index=False))

conn.close()
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
