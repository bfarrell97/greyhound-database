"""
PROFITABLE ANGLE HUNTING - Looking for positive ROI opportunities
Based on the data analysis findings
"""
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('greyhound_racing.db')

print("="*80)
print("HUNTING FOR PROFITABLE ANGLES")
print("="*80)

# KEY INSIGHTS FROM PREVIOUS ANALYSIS:
# 1. All price bands show NEGATIVE edge vs market (market is efficient)
# 2. Box 1 has 19.2% win rate (vs 12.5% random) but market prices it correctly
# 3. Heavy dogs (32-34kg) win 15.2% vs light (<26kg) 12.3%
# 4. Split benchmark is INVERTED - slow early = MORE wins (counterintuitive)
# 5. Some trainers at specific tracks have 45-55% win rates

# Let's look for OUTLIER situations where market might be wrong

print("\n" + "="*80)
print("1. TRAINER SPECIALISTS AT SPECIFIC TRACKS - ROI ANALYSIS")
print("="*80)

trainer_track_roi = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CASE WHEN ge.StartingPrice != '' THEN CAST(ge.StartingPrice AS REAL) END) as avg_sp,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' AND CAST(ge.StartingPrice AS REAL) > 1
    GROUP BY tr.TrainerName, t.TrackName
    HAVING COUNT(*) >= 50
    ORDER BY roi DESC
    LIMIT 30
""", conn)
print(trainer_track_roi.to_string(index=False))

# 2. Look at Box 1 at tracks with EXTREME box bias
print("\n" + "="*80)
print("2. BOX 1 AT HIGH-BIAS TRACKS (where box 1 has >20% win rate)")
print("="*80)

box1_tracks_roi = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CAST(ge.StartingPrice AS REAL)) as avg_sp,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Box = 1 
      AND ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 10
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 500
    ORDER BY roi DESC
    LIMIT 20
""", conn)
print(box1_tracks_roi.to_string(index=False))

# 3. Heavy dogs at short distances (potential physical advantage)
print("\n" + "="*80)
print("3. HEAVY DOGS (32kg+) AT SHORT SPRINTS (<350m)")
print("="*80)

heavy_short_roi = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN Weight >= 34 THEN 'Very Heavy 34kg+'
            WHEN Weight >= 32 THEN 'Heavy 32-34kg'
            WHEN Weight >= 30 THEN 'Medium 30-32kg'
            ELSE 'Light <30kg'
        END as weight_cat,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 10
      AND r.Distance < 350
      AND ge.Weight IS NOT NULL
    GROUP BY weight_cat
    ORDER BY weight_cat
""", conn)
print(heavy_short_roi.to_string(index=False))

# 4. Grade drops - dogs stepping DOWN in class
print("\n" + "="*80)
print("4. FAVORITES AT SPECIFIC GRADES (odds $1.50-$3.00)")
print("="*80)

grade_fav_roi = pd.read_sql_query("""
    SELECT 
        r.Grade,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 3.0
      AND r.Grade IS NOT NULL AND r.Grade != ''
    GROUP BY r.Grade
    HAVING COUNT(*) >= 500
    ORDER BY roi DESC
    LIMIT 20
""", conn)
print(grade_fav_roi.to_string(index=False))

# 5. Race number patterns
print("\n" + "="*80)
print("5. EARLY VS LATE RACES (Favorites $1.50-$3)")
print("="*80)

race_num_roi = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN r.RaceNumber <= 3 THEN 'Early (1-3)'
            WHEN r.RaceNumber <= 6 THEN 'Mid-Early (4-6)'
            WHEN r.RaceNumber <= 9 THEN 'Mid-Late (7-9)'
            ELSE 'Late (10+)'
        END as race_position,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 3.0
    GROUP BY race_position
    ORDER BY race_position
""", conn)
print(race_num_roi.to_string(index=False))

# 6. COMPLEX: Combine multiple factors - trainer + track + price range
print("\n" + "="*80)
print("6. BEST TRAINER+TRACK COMBOS AT VALUE PRICES ($3-$7)")
print("="*80)

combo_value = pd.read_sql_query("""
    SELECT 
        tr.TrainerName,
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 3.0 AND 7.0
    GROUP BY tr.TrainerName, t.TrackName
    HAVING COUNT(*) >= 30 AND SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) >= 5
    ORDER BY roi DESC
    LIMIT 30
""", conn)
print(combo_value.to_string(index=False))

# 7. Box 1 + Favorite at specific tracks
print("\n" + "="*80)
print("7. BOX 1 FAVORITES ($1.50-$3) AT BEST TRACKS")
print("="*80)

box1_fav = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Box = 1 
      AND ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 1.5 AND 3.0
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 200
    ORDER BY roi DESC
    LIMIT 15
""", conn)
print(box1_fav.to_string(index=False))

# 8. Look at LONGSHOTS that beat the market
print("\n" + "="*80)
print("8. LONGSHOT VALUE ($10-$20) BY TRACK")
print("="*80)

longshot_roi = pd.read_sql_query("""
    SELECT 
        t.TrackName,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        ROUND(100.0 / AVG(CAST(ge.StartingPrice AS REAL)), 2) as implied_win_pct,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) BETWEEN 10 AND 20
    GROUP BY t.TrackName
    HAVING COUNT(*) >= 500
    ORDER BY roi DESC
    LIMIT 15
""", conn)
print(longshot_roi.to_string(index=False))

# 9. Split benchmark outliers - dogs with BEST early speed
print("\n" + "="*80)
print("9. FASTEST EARLY SPEED (SplitBenchmark < -3) AT SPRINTS")
print("="*80)

fast_early = pd.read_sql_query("""
    SELECT 
        CASE 
            WHEN SplitBenchmarkLengths < -4 THEN 'Elite (<-4)'
            WHEN SplitBenchmarkLengths < -3 THEN 'Excellent (-4 to -3)'
            WHEN SplitBenchmarkLengths < -2 THEN 'Very Good (-3 to -2)'
            ELSE 'Good (-2 to -1)'
        END as early_speed,
        COUNT(*) as bets,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_pct,
        AVG(CAST(ge.StartingPrice AS REAL)) as avg_sp,
        SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) as profit,
        ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN CAST(ge.StartingPrice AS REAL) - 1 ELSE -1 END) / COUNT(*), 2) as roi
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    WHERE ge.SplitBenchmarkLengths IS NOT NULL
      AND ge.SplitBenchmarkLengths < -1
      AND ge.StartingPrice != '' 
      AND CAST(ge.StartingPrice AS REAL) > 1
      AND r.Distance < 400
    GROUP BY early_speed
    ORDER BY MIN(ge.SplitBenchmarkLengths)
""", conn)
print(fast_early.to_string(index=False))

conn.close()

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print("""
KEY TAKEAWAYS:

1. MARKET IS VERY EFFICIENT
   - Almost all combinations show negative ROI
   - The market correctly prices box bias, weight, etc.

2. POTENTIAL EDGES (need more investigation):
   - Some trainer+track combinations show positive ROI at value prices
   - Specific tracks may have slight mispricing in longshot range
   - Early speed benchmark may have predictive value not fully priced

3. WHAT DOESN'T WORK:
   - Blindly betting favorites (negative ROI at all price points)
   - Box bias alone (market prices it correctly)
   - Weight-based strategies (market knows this)

4. NEXT STEPS:
   - Focus on trainer+track specialists at value prices
   - Look for dogs with improving early speed (split benchmark)
   - Consider track-specific models
   - Time-based analysis (is data from 2024-2025 different from 2020-2022?)
""")
