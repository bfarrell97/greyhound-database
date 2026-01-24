"""
FINAL ANALYSIS: Actionable Betting Angles
Focus on the most promising patterns from the data
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('greyhound_racing.db')

print("="*80)
print("ACTIONABLE BETTING ANGLES - FINAL ANALYSIS")
print("="*80)

# The trainer+track combos at $3-$7 showed the most promise
# Let's validate these with more stringent requirements

print("\n" + "="*80)
print("1. TRAINER+TRACK SPECIALISTS (stricter: min 50 bets, 15+ wins)")
print("   Price range: $3.00 - $10.00 (value range)")
print("="*80)

trainer_value = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CAST(ge.StartingPrice AS REAL)) as avg_sp,
        ROUND(100.0 / AVG(CAST(ge.StartingPrice AS REAL)), 2) as implied_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 3.0 AND 10.0
    GROUP BY tr.TrainerName, t.TrackName
    HAVING COUNT(*) >= 50 AND SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) >= 15
    ORDER BY roi DESC
    LIMIT 20
""", conn)
print(trainer_value.to_string(index=False))

# Check if these trainers are still active (recent data)
print("\n" + "="*80)
print("2. ARE THESE TRAINERS STILL ACTIVE? (2025 data)")
print("="*80)

recent_trainers = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        t.TrackName,
        COUNT(*) as bets_2025,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins_2025,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct_2025,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit_2025,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi_2025
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 3.0 AND 10.0
      AND rm.MeetingDate >= '2025-01-01'
      AND (tr.TrainerName, t.TrackName) IN (
          SELECT tr2.TrainerName, t2.TrackName
          FROM GreyhoundEntries ge2
          JOIN Trainers tr2 ON ge2.TrainerID = tr2.TrainerID
          JOIN Races r2 ON ge2.RaceID = r2.RaceID
          JOIN RaceMeetings rm2 ON r2.MeetingID = rm2.MeetingID
          JOIN Tracks t2 ON rm2.TrackID = t2.TrackID
          WHERE ge2.StartingPrice != '' 
            AND CAST(ge2.StartingPrice AS REAL) BETWEEN 3.0 AND 10.0
          GROUP BY tr2.TrainerName, t2.TrackName
          HAVING COUNT(*) >= 50 
            AND SUM(CASE WHEN ge2.Position = '1' THEN 1 ELSE 0 END) >= 15
            AND 100.0 * SUM(CASE WHEN ge2.Position = '1' THEN CAST(ge2.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*) > 20
      )
    GROUP BY tr.TrainerName, t.TrackName
    HAVING COUNT(*) >= 5
    ORDER BY profit_2025 DESC
""", conn)
print(recent_trainers.to_string(index=False))

# Look at dogs returning to tracks where they've won before
print("\n" + "="*80)
print("3. TRACK SPECIALISTS: Dogs returning to winning tracks")
print("   (Dogs with 2+ wins at a track, running there again)")
print("="*80)

# This would require a more complex query - let's look at a proxy
# Dogs with multiple wins at same track in 2025
track_specialists = pd.read_sql_query("""
    WITH dog_track_wins AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            t.TrackName,
            t.TrackID,
            COUNT(*) as runs_at_track,
            SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins_at_track
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        GROUP BY ge.GreyhoundID, g.GreyhoundName, t.TrackName, t.TrackID
        HAVING runs_at_track >= 5 AND wins_at_track >= 2
    )
    SELECT 
        runs_at_track,
        wins_at_track,
        COUNT(*) as dog_track_combos,
        ROUND(100.0 * AVG(wins_at_track * 1.0 / runs_at_track), 2) as avg_win_rate_at_track
    FROM dog_track_wins
    GROUP BY runs_at_track, wins_at_track
    HAVING COUNT(*) >= 100
    ORDER BY wins_at_track DESC, runs_at_track
    LIMIT 20
""", conn)
print(track_specialists.to_string(index=False))

# Look at the split benchmark trend - improving early speed
print("\n" + "="*80)
print("4. IMPROVING EARLY SPEED DOGS")
print("   (Average split benchmark improving in last 5 races)")
print("="*80)

# Check if there's data to support this
split_trend = pd.read_sql_query("""
    WITH dog_splits AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            rm.MeetingDate,
            ge.SplitBenchmarkLengths,
            ge.Position,
            ge.StartingPrice,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as race_num
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.SplitBenchmarkLengths IS NOT NULL
          AND ge.StartingPrice IS NOT NULL
          AND ge.StartingPrice != ''
    ),
    dog_split_trend AS (
        SELECT 
            GreyhoundID,
            GreyhoundName,
            AVG(CASE WHEN race_num <= 3 THEN SplitBenchmarkLengths END) as recent_split,
            AVG(CASE WHEN race_num BETWEEN 4 AND 6 THEN SplitBenchmarkLengths END) as older_split,
            COUNT(CASE WHEN race_num <= 3 THEN 1 END) as recent_count,
            COUNT(CASE WHEN race_num BETWEEN 4 AND 6 THEN 1 END) as older_count
        FROM dog_splits
        GROUP BY GreyhoundID, GreyhoundName
        HAVING recent_count >= 3 AND older_count >= 3
    )
    SELECT 
        CASE 
            WHEN recent_split < older_split - 0.5 THEN 'Improving (>0.5L faster)'
            WHEN recent_split < older_split THEN 'Slight Improve'
            WHEN recent_split < older_split + 0.5 THEN 'Stable'
            ELSE 'Declining'
        END as trend,
        COUNT(*) as dogs,
        ROUND(AVG(recent_split), 2) as avg_recent_split,
        ROUND(AVG(older_split), 2) as avg_older_split
    FROM dog_split_trend
    GROUP BY trend
    ORDER BY AVG(recent_split - older_split)
""", conn)
print(split_trend.to_string(index=False))

# Look at class drop - dogs dropping in grade
print("\n" + "="*80)  
print("5. GRADE/CLASS PATTERNS")
print("   Looking at performance by grade at different price points")
print("="*80)

# Maidens at value prices might be underpriced
maiden_value = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN r.Grade LIKE '%Maiden%' THEN 'Maiden'
            WHEN r.Grade LIKE '%Grade 7%' OR r.Grade LIKE '%Grade 6%' THEN 'Low Grade (6-7)'
            WHEN r.Grade LIKE '%Grade 5%' THEN 'Grade 5'
            WHEN r.Grade LIKE '%Grade 4%' OR r.Grade LIKE '%Grade 3%' THEN 'Mid Grade (3-4)'
            WHEN r.Grade LIKE '%Free For All%' OR r.Grade LIKE '%Open%' THEN 'Open/FFA'
            ELSE 'Other'
        END as grade_group,
        CASE 
            WHEN CAST(ge.StartingPrice AS REAL) < 3 THEN 'Fav <$3'
            WHEN CAST(ge.StartingPrice AS REAL) < 5 THEN 'Value $3-$5'
            WHEN CAST(ge.StartingPrice AS REAL) < 10 THEN 'Outsider $5-$10'
            ELSE 'Longshot $10+'
        END as price_band,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) > 1
      AND r.Grade IS NOT NULL
    GROUP BY grade_group, price_band
    HAVING COUNT(*) >= 5000
    ORDER BY grade_group, price_band
""", conn)
print(maiden_value.to_string(index=False))

# Final: Best overall filters
print("\n" + "="*80)
print("6. COMBINING FACTORS: Box 1 + Short Distance + Grade 5-7")
print("="*80)

combo_filter = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN CAST(ge.StartingPrice AS REAL) < 2 THEN '$1.50-$2.00'
            WHEN CAST(ge.StartingPrice AS REAL) < 3 THEN '$2.00-$3.00'
            WHEN CAST(ge.StartingPrice AS REAL) < 5 THEN '$3.00-$5.00'
            ELSE '$5.00-$10.00'
        END as price_band,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.Box = 1
      AND r.Distance < 400
      AND (r.Grade LIKE '%Grade 5%' OR r.Grade LIKE '%Grade 6%' OR r.Grade LIKE '%Grade 7%')
      AND ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 10
    GROUP BY price_band
    ORDER BY MIN(CAST(ge.StartingPrice AS REAL))
""", conn)
print(combo_filter.to_string(index=False))

conn.close()

print("\n" + "="*80)
print("FINAL CONCLUSIONS")
print("="*80)
print("""
REALITY CHECK:
==============
After analyzing ~2 million race entries across 6 years (2020-2025):

1. THE MARKET IS VERY EFFICIENT
   - All simple angle (box, weight, grade, track) show negative ROI
   - The market correctly prices most factors we can observe

2. SMALL POTENTIAL EDGES (high variance, likely overfitting):
   - Specific trainer+track combinations at value prices ($3-$10)
   - Some trainers show +50% to +100% ROI but with small sample sizes
   - These could be real edges OR random variance

3. WHAT YOU'D NEED FOR A REAL EDGE:
   - Inside information (track conditions, dog health, trainer intent)
   - Real-time data (scratchings, market moves, late money)
   - Proprietary speed ratings with better track adjustment
   - Machine learning on raw run data (not just summary stats)

4. THE HONEST ASSESSMENT:
   - Greyhound racing markets in Australia are mature and efficient
   - The ~15-20% takeout makes it very hard to profit long-term
   - Even the "best" angles we found are likely noise, not signal
   
5. IF YOU STILL WANT TO TRY:
   - Focus on trainer specialists at their home tracks
   - Look for value prices ($3-$10), not favorites
   - Track your bets meticulously to identify real patterns
   - Be prepared to lose money - treat it as entertainment
""")
