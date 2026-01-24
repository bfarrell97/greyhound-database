"""
Beyer Speed Figure Generator
=============================
Calculates Beyer-style speed figures for greyhound races.

Formula: BSF = Base + ((TrackPar - ActualTime) × ScaleFactor)
- Base: 100 (average speed)
- TrackPar: Median time for track/distance (from Benchmarks)
- ScaleFactor: Points per length (~0.05 seconds per length at 500m)

Higher = Faster
100 = Par for track/distance
110 = 10 points faster than par
"""
import sqlite3
import pandas as pd
import numpy as np

# Scale factor: approximately 0.05 seconds per length, 3 points per length
POINTS_PER_SECOND = 60  # 3 points / 0.05 sec = 60 points per second
BASE_FIGURE = 100

def calculate_beyer_figures():
    print("="*70)
    print("BEYER SPEED FIGURE GENERATOR")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # First, add column if not exists
    print("\n[1/4] Adding BeyerSpeedFigure column...")
    try:
        conn.execute("ALTER TABLE GreyhoundEntries ADD COLUMN BeyerSpeedFigure REAL")
        print("  Column added")
    except:
        print("  Column already exists")
    
    # Load benchmarks
    print("\n[2/4] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    print(f"  Loaded {len(bench_lookup)} track/distance benchmarks")
    
    # Load entries without BeyerSpeedFigure
    print("\n[3/4] Loading entries...")
    query = """
    SELECT ge.EntryID, ge.FinishTime, r.Distance, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTime IS NOT NULL 
      AND ge.FinishTime > 0
      AND ge.BeyerSpeedFigure IS NULL
    """
    df = pd.read_sql_query(query, conn)
    print(f"  Found {len(df):,} entries to process")
    
    if len(df) == 0:
        print("\n  All entries already have Beyer figures!")
        conn.close()
        return
    
    print("\n[4/4] Calculating Beyer Speed Figures...")
    
    updates = []
    processed = 0
    no_benchmark = 0
    
    for _, row in df.iterrows():
        track = row['TrackName']
        distance = row['Distance']
        finish_time = row['FinishTime']
        entry_id = row['EntryID']
        
        # Get benchmark
        par_time = bench_lookup.get((track, distance))
        
        if par_time is not None and finish_time > 0:
            # Calculate Beyer figure
            # Negative diff = faster than par = higher figure
            time_diff = par_time - finish_time
            beyer = BASE_FIGURE + (time_diff * POINTS_PER_SECOND)
            
            # Cap at reasonable range (50-150)
            beyer = max(50, min(150, beyer))
            
            updates.append((beyer, entry_id))
        else:
            no_benchmark += 1
        
        processed += 1
        if processed % 100000 == 0:
            print(f"  {processed:,} processed...")
    
    print(f"  Total processed: {processed:,}")
    print(f"  With figures: {len(updates):,}")
    print(f"  No benchmark: {no_benchmark:,}")
    
    # Update database
    if updates:
        print(f"\n  Updating database with {len(updates):,} figures...")
        conn.executemany("UPDATE GreyhoundEntries SET BeyerSpeedFigure = ? WHERE EntryID = ?", updates)
        conn.commit()
        print("  Done!")
    
    # Show distribution
    print("\n" + "="*70)
    print("DISTRIBUTION")
    print("="*70)
    
    stats_df = pd.read_sql_query("""
        SELECT BeyerSpeedFigure, COUNT(*) as Count
        FROM GreyhoundEntries
        WHERE BeyerSpeedFigure IS NOT NULL
        GROUP BY ROUND(BeyerSpeedFigure / 10) * 10
        ORDER BY BeyerSpeedFigure
    """, conn)
    
    for _, row in stats_df.iterrows():
        bucket = int(row['BeyerSpeedFigure'] // 10) * 10
        count = row['Count']
        bar = '█' * min(50, count // 10000)
        print(f"  {bucket:3d}-{bucket+9}: {count:8,} {bar}")
    
    # Show average by grade
    print("\n" + "="*70)
    print("AVERAGE BY GRADE")
    print("="*70)
    
    grade_df = pd.read_sql_query("""
        SELECT ge.IncomingGrade, 
               AVG(ge.BeyerSpeedFigure) as AvgBSF,
               COUNT(*) as Count
        FROM GreyhoundEntries ge
        WHERE ge.BeyerSpeedFigure IS NOT NULL
          AND ge.IncomingGrade IS NOT NULL
        GROUP BY ge.IncomingGrade
        ORDER BY AvgBSF DESC
    """, conn)
    
    print(f"{'Grade':<12} {'Avg BSF':>8} {'Count':>10}")
    print("-" * 32)
    for _, row in grade_df.iterrows():
        print(f"{str(row['IncomingGrade']):<12} {row['AvgBSF']:8.1f} {int(row['Count']):10,}")
    
    conn.close()
    
    print("\n" + "="*70)
    print("COMPLETE - BeyerSpeedFigure column populated")
    print("="*70)

if __name__ == "__main__":
    calculate_beyer_figures()
