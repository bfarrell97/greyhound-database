import sqlite3
import os

DB_PATH = 'greyhound_racing.db'

def wipe_today():
    today = '2025-12-26'
    print(f"--- WIPING ALL DATA FOR {today} ---")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Get MeetingIDs for today
        cursor.execute("SELECT MeetingID FROM RaceMeetings WHERE MeetingDate = ?", (today,))
        mids = [row[0] for row in cursor.fetchall()]
        
        if not mids:
            print(f"[INFO] No meetings found for {today}. Nothing to wipe.")
            return

        print(f"Found {len(mids)} meetings. Deleting related entries...")

        # 2. Delete Entries
        cursor.execute(f"""
            DELETE FROM GreyhoundEntries 
            WHERE RaceID IN (
                SELECT RaceID FROM Races WHERE MeetingID IN ({','.join(['?']*len(mids))})
            )
        """, mids)
        print(f"  Deleted {cursor.rowcount} GreyhoundEntries.")

        # 3. Delete Races
        cursor.execute(f"DELETE FROM Races WHERE MeetingID IN ({','.join(['?']*len(mids))})", mids)
        print(f"  Deleted {cursor.rowcount} Races.")

        # 4. Delete Meetings
        cursor.execute("DELETE FROM RaceMeetings WHERE MeetingDate = ?", (today,))
        print(f"  Deleted {cursor.rowcount} RaceMeetings.")

        conn.commit()
        print("\n[OK] Database cleaned. All corrupted entries for today have been removed.")
        print("[TIP] Restart the App and click 'Load Today's Tips' to see the fresh Betfair-direct data.")

    except Exception as e:
        print(f"[ERROR] Wipe failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    wipe_today()
