"""
Fast Meeting Benchmark Calculator
Populates MeetingAvgBenchmarkLengths and MeetingSplitAvgBenchmarkLengths
using efficient SQL aggregation
"""

import sqlite3
import time

DB_PATH = 'greyhound_racing.db'

def populate_meeting_benchmarks():
    print("="*60)
    print("FAST MEETING BENCHMARK CALCULATOR")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    
    # Check current status
    cursor.execute("""
        SELECT COUNT(*), 
               SUM(CASE WHEN MeetingAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN MeetingSplitAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END)
        FROM RaceMeetings
    """)
    total, has_avg, has_split = cursor.fetchone()
    print(f"\nBefore: {total:,} meetings")
    print(f"  MeetingAvgBenchmarkLengths: {has_avg:,} ({has_avg/total*100:.1f}%)")
    print(f"  MeetingSplitAvgBenchmarkLengths: {has_split:,} ({has_split/total*100:.1f}%)")
    
    # Step 1: Calculate MeetingAvgBenchmarkLengths (average of FinishTimeBenchmarkLengths per meeting)
    print("\n[1/2] Calculating MeetingAvgBenchmarkLengths...")
    start = time.time()
    
    cursor.execute("""
        UPDATE RaceMeetings
        SET MeetingAvgBenchmarkLengths = (
            SELECT AVG(ge.FinishTimeBenchmarkLengths)
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.MeetingID = RaceMeetings.MeetingID
              AND ge.FinishTimeBenchmarkLengths IS NOT NULL
        )
        WHERE MeetingAvgBenchmarkLengths IS NULL
    """)
    updated1 = cursor.rowcount
    conn.commit()
    print(f"  Updated {updated1:,} meetings in {time.time()-start:.1f}s")
    
    # Step 2: Calculate MeetingSplitAvgBenchmarkLengths (average of SplitBenchmarkLengths per meeting)
    print("\n[2/2] Calculating MeetingSplitAvgBenchmarkLengths...")
    start = time.time()
    
    cursor.execute("""
        UPDATE RaceMeetings
        SET MeetingSplitAvgBenchmarkLengths = (
            SELECT AVG(ge.SplitBenchmarkLengths)
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.MeetingID = RaceMeetings.MeetingID
              AND ge.SplitBenchmarkLengths IS NOT NULL
        )
        WHERE MeetingSplitAvgBenchmarkLengths IS NULL
    """)
    updated2 = cursor.rowcount
    conn.commit()
    print(f"  Updated {updated2:,} meetings in {time.time()-start:.1f}s")
    
    # Check final status
    cursor.execute("""
        SELECT COUNT(*), 
               SUM(CASE WHEN MeetingAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN MeetingSplitAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END)
        FROM RaceMeetings
    """)
    total, has_avg, has_split = cursor.fetchone()
    print(f"\nAfter: {total:,} meetings")
    print(f"  MeetingAvgBenchmarkLengths: {has_avg:,} ({has_avg/total*100:.1f}%)")
    print(f"  MeetingSplitAvgBenchmarkLengths: {has_split:,} ({has_split/total*100:.1f}%)")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    populate_meeting_benchmarks()
