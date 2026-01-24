
import sqlite3
import os
import sys

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.import_bsp import apply_updates

DB_PATH = 'greyhound_racing.db'

def verify_place_support():
    print("Verifying Place Price Support...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Verify schema
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    cols = [row[1] for row in cursor.fetchall()]
    if 'BSPPlace' in cols:
        print("[OK] BSPPlace column exists.")
    else:
        print("[FAIL] BSPPlace column missing!")
        return

    # 2. Mock a result for a known race in the DB
    # We'll pick one entry and try to update its BSPPlace
    cursor.execute("""
        SELECT ge.EntryID, g.GreyhoundName, t.TrackName, rm.MeetingDate, r.RaceNumber
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        ORDER BY rm.MeetingDate DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        print("[SKIP] No data in DB to test with.")
        return
    
    entry_id, dog_name, track, date, race_num = row
    print(f"Testing with: {dog_name} at {track} R{race_num} on {date}")
    
    # Reset BSPPlace for test
    cursor.execute("UPDATE GreyhoundEntries SET BSPPlace = NULL WHERE EntryID = ?", (entry_id,))
    conn.commit()
    
    # Test apply_updates with PLACE marketType
    mock_updates = [(track, date, race_num, dog_name, 5.5, 'PLACE')]
    apply_updates(conn, cursor, mock_updates)
    
    # Verify update
    cursor.execute("SELECT BSPPlace FROM GreyhoundEntries WHERE EntryID = ?", (entry_id,))
    val = cursor.fetchone()[0]
    
    if val == 5.5:
        print("[OK] apply_updates correctly updated BSPPlace for PLACE market.")
    else:
        print(f"[FAIL] BSPPlace was {val}, expected 5.5")
        
    conn.close()

if __name__ == "__main__":
    verify_place_support()
