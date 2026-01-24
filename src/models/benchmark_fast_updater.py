"""
FAST Benchmark Comparison Calculator
Uses bulk SQL UPDATE instead of row-by-row processing
1 length = 0.07 seconds
Positive values = faster than benchmark (better)
Negative values = slower than benchmark (worse)
"""

import sqlite3
import time

SECONDS_PER_LENGTH = 0.07

def calculate_benchmark_comparisons_fast(progress_callback=None):
    """Calculate benchmark comparisons using fast bulk SQL updates"""

    db_path = 'greyhound_racing.db'
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    start_time = time.time()

    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    log("=" * 70)
    log("FAST BENCHMARK COMPARISON CALCULATOR")
    log("=" * 70)
    log(f"1 length = {SECONDS_PER_LENGTH} seconds")
    log("Positive = faster than benchmark, Negative = slower")
    log("")

    # Step 1: Calculate track/distance benchmarks from ALL winning times
    log("Step 1: Creating/updating track benchmarks...")
    step1_start = time.time()
    
    # Clear existing benchmarks and recalculate
    cursor.execute("DELETE FROM Benchmarks")
    
    cursor.execute("""
        INSERT INTO Benchmarks (TrackName, Distance, AvgTime, MedianTime, FastestTime, 
                                 SlowestTime, StdDev, SampleSize, AvgSplit, SplitSampleSize)
        SELECT 
            t.TrackName,
            r.Distance,
            AVG(ge.FinishTime) as AvgTime,
            AVG(ge.FinishTime) as MedianTime,  -- SQLite doesn't have MEDIAN, using AVG
            MIN(ge.FinishTime) as FastestTime,
            MAX(ge.FinishTime) as SlowestTime,
            0 as StdDev,
            COUNT(*) as SampleSize,
            AVG(ge.Split) as AvgSplit,
            SUM(CASE WHEN ge.Split IS NOT NULL THEN 1 ELSE 0 END) as SplitSampleSize
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.FinishTime IS NOT NULL
          AND ge.Position = '1'
        GROUP BY t.TrackName, r.Distance
        HAVING COUNT(*) >= 5
    """)
    
    benchmark_count = cursor.rowcount
    conn.commit()
    log(f"  Created {benchmark_count} track/distance benchmarks in {time.time()-step1_start:.1f}s")

    # Step 2: Bulk update FinishTimeBenchmarkLengths for all entries
    log("\nStep 2: Updating entry finish time benchmarks (bulk SQL)...")
    step2_start = time.time()
    
    cursor.execute("""
        UPDATE GreyhoundEntries
        SET FinishTimeBenchmarkLengths = (
            SELECT (b.AvgTime - GreyhoundEntries.FinishTime) / ?
            FROM Benchmarks b
            JOIN Races r ON GreyhoundEntries.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE b.TrackName = t.TrackName
              AND b.Distance = r.Distance
        )
        WHERE FinishTime IS NOT NULL
    """, (SECONDS_PER_LENGTH,))
    
    finish_updated = cursor.rowcount
    conn.commit()
    log(f"  Updated {finish_updated:,} finish benchmarks in {time.time()-step2_start:.1f}s")

    # Step 3: Bulk update SplitBenchmarkLengths for all entries
    log("\nStep 3: Updating entry split benchmarks (bulk SQL)...")
    step3_start = time.time()
    
    cursor.execute("""
        UPDATE GreyhoundEntries
        SET SplitBenchmarkLengths = (
            SELECT (b.AvgSplit - GreyhoundEntries.Split) / ?
            FROM Benchmarks b
            JOIN Races r ON GreyhoundEntries.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE b.TrackName = t.TrackName
              AND b.Distance = r.Distance
              AND b.AvgSplit IS NOT NULL
        )
        WHERE Split IS NOT NULL
    """, (SECONDS_PER_LENGTH,))
    
    split_updated = cursor.rowcount
    conn.commit()
    log(f"  Updated {split_updated:,} split benchmarks in {time.time()-step3_start:.1f}s")

    # Step 4: Bulk update meeting-level averages
    # Step 4: Bulk update meeting-level averages using TEMP TABLE for speed
    log("\nStep 4: Updating meeting averages (Temp Table Optimization)...")
    step4_start = time.time()
    
    # 1. Calculate averages into a temporary table (One scan of GreyhoundEntries)
    log("  Calculating averages...")
    cursor.execute("DROP TABLE IF EXISTS TempMeetingStats")
    cursor.execute("""
        CREATE TEMP TABLE TempMeetingStats AS
        SELECT 
            r.MeetingID,
            AVG(ge.FinishTimeBenchmarkLengths) as AvgFinish,
            AVG(ge.SplitBenchmarkLengths) as AvgSplit
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL 
           OR ge.SplitBenchmarkLengths IS NOT NULL
        GROUP BY r.MeetingID
    """)
    
    # 2. Add index for fast lookups
    cursor.execute("CREATE INDEX idx_temp_meeting ON TempMeetingStats(MeetingID)")
    log(f"  Calculated stats for {cursor.rowcount} meetings in temp table in {time.time()-step4_start:.1f}s")
    
    # 3. Update the main table using the temp table
    log("  Applying updates to RaceMeetings...")
    update_start = time.time()
    cursor.execute("""
        UPDATE RaceMeetings
        SET 
            MeetingAvgBenchmarkLengths = (
                SELECT AvgFinish 
                FROM TempMeetingStats 
                WHERE TempMeetingStats.MeetingID = RaceMeetings.MeetingID
            ),
            MeetingSplitAvgBenchmarkLengths = (
                SELECT AvgSplit 
                FROM TempMeetingStats 
                WHERE TempMeetingStats.MeetingID = RaceMeetings.MeetingID
            )
        WHERE EXISTS (
            SELECT 1 FROM TempMeetingStats WHERE TempMeetingStats.MeetingID = RaceMeetings.MeetingID
        )
    """)
    
    meetings_updated = cursor.rowcount
    
    # 4. Cleanup
    cursor.execute("DROP TABLE TempMeetingStats")
    conn.commit()
    log(f"  Updated {meetings_updated:,} meeting averages in {time.time()-update_start:.1f}s")

    # Step 5: Summary statistics
    log("\nStep 5: Verification...")
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN FinishTimeBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as has_finish,
            SUM(CASE WHEN SplitBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as has_split,
            AVG(FinishTimeBenchmarkLengths) as avg_finish_bench,
            AVG(SplitBenchmarkLengths) as avg_split_bench
        FROM GreyhoundEntries
    """)
    
    row = cursor.fetchone()
    log(f"  Total entries: {row[0]:,}")
    log(f"  With finish benchmark: {row[1]:,} ({row[1]/row[0]*100:.1f}%)")
    log(f"  With split benchmark: {row[2]:,} ({row[2]/row[0]*100:.1f}%)")
    log(f"  Avg finish benchmark: {row[3]:.2f} lengths" if row[3] else "  Avg finish benchmark: N/A")
    log(f"  Avg split benchmark: {row[4]:.2f} lengths" if row[4] else "  Avg split benchmark: N/A")
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN MeetingAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as has_meeting_avg
        FROM RaceMeetings
    """)
    
    row = cursor.fetchone()
    log(f"  Meetings with avg benchmark: {row[1]:,}/{row[0]:,} ({row[1]/row[0]*100:.1f}%)")

    total_time = time.time() - start_time
    log("")
    log("=" * 70)
    log(f"COMPLETED in {total_time:.1f} seconds")
    log("=" * 70)

    conn.close()
    return benchmark_count, finish_updated, split_updated, meetings_updated


if __name__ == "__main__":
    calculate_benchmark_comparisons_fast()
