"""
Fast meeting benchmark calculator with progress updates
"""
import sqlite3
import time

DB_PATH = 'greyhound_racing.db'

def main():
    print("="*60)
    print("FAST MEETING BENCHMARK CALCULATOR")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    
    # Get all meetings that need updating
    print("\nFinding meetings to update...")
    cursor = conn.execute("""
        SELECT MeetingID FROM RaceMeetings 
        WHERE MeetingAvgBenchmarkLengths IS NULL 
           OR MeetingSplitAvgBenchmarkLengths IS NULL
    """)
    meetings = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(meetings)} meetings to update")
    
    if len(meetings) == 0:
        print("All meetings already have benchmarks!")
        conn.close()
        return
    
    # Process in batches
    batch_size = 500
    updated = 0
    start = time.time()
    
    for i in range(0, len(meetings), batch_size):
        batch = meetings[i:i+batch_size]
        
        # Calculate averages for this batch using a single query
        placeholders = ','.join('?' * len(batch))
        
        cursor = conn.execute(f"""
            SELECT 
                r.MeetingID,
                AVG(ge.FinishTimeBenchmarkLengths) as avg_finish,
                AVG(ge.SplitBenchmarkLengths) as avg_split
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.MeetingID IN ({placeholders})
              AND ge.FinishTimeBenchmarkLengths IS NOT NULL
            GROUP BY r.MeetingID
        """, batch)
        
        results = cursor.fetchall()
        
        # Update each meeting
        for meeting_id, avg_finish, avg_split in results:
            conn.execute("""
                UPDATE RaceMeetings 
                SET MeetingAvgBenchmarkLengths = ?,
                    MeetingSplitAvgBenchmarkLengths = ?
                WHERE MeetingID = ?
            """, (avg_finish, avg_split, meeting_id))
        
        conn.commit()
        updated += len(results)
        
        # Progress update
        elapsed = time.time() - start
        pct = (i + len(batch)) / len(meetings) * 100
        rate = updated / elapsed if elapsed > 0 else 0
        remaining = (len(meetings) - i - len(batch)) / rate if rate > 0 else 0
        
        print(f"  {i + len(batch):,}/{len(meetings):,} ({pct:.1f}%) - {updated:,} updated - {remaining:.0f}s remaining")
    
    conn.close()
    
    print(f"\nDone! Updated {updated:,} meetings in {time.time()-start:.1f}s")
    
    # Verify
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN MeetingAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as has_avg,
            SUM(CASE WHEN MeetingSplitAvgBenchmarkLengths IS NOT NULL THEN 1 ELSE 0 END) as has_split
        FROM RaceMeetings
    """)
    row = cursor.fetchone()
    print(f"\nMeeting coverage now:")
    print(f"  MeetingAvgBenchmarkLengths: {row[1]:,}/{row[0]:,} ({row[1]/row[0]*100:.1f}%)")
    print(f"  MeetingSplitAvgBenchmarkLengths: {row[2]:,}/{row[0]:,} ({row[2]/row[0]*100:.1f}%)")
    conn.close()

if __name__ == "__main__":
    main()
