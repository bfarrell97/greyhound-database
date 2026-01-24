"""Delete non-TAB track entries from database"""
import sqlite3

DB_PATH = 'greyhound_racing.db'

# Non-TAB tracks to delete
NON_TAB_TRACKS = [
    'Coonamble',
    'Tamworth', 
    'Young',
    'Potts Park',
    'Lithgow',
    'Moree',
    'Kempsey',
    'Coonabarabran',
    'Capalaba'  # The old one (55 entries), not Bet Deluxe Capalaba
]

conn = sqlite3.connect(DB_PATH)

print("="*60)
print("DELETING NON-TAB TRACK ENTRIES")
print("="*60)

for track in NON_TAB_TRACKS:
    # Get track ID
    cursor = conn.execute("SELECT TrackID FROM Tracks WHERE TrackName = ?", (track,))
    result = cursor.fetchone()
    
    if not result:
        print(f"{track}: Not found in Tracks table")
        continue
    
    track_id = result[0]
    
    # Count entries
    cursor = conn.execute("""
        SELECT COUNT(*) FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.TrackID = ?
    """, (track_id,))
    entry_count = cursor.fetchone()[0]
    
    # Delete entries
    conn.execute("""
        DELETE FROM GreyhoundEntries WHERE RaceID IN (
            SELECT r.RaceID FROM Races r
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE rm.TrackID = ?
        )
    """, (track_id,))
    
    # Delete races
    conn.execute("""
        DELETE FROM Races WHERE MeetingID IN (
            SELECT MeetingID FROM RaceMeetings WHERE TrackID = ?
        )
    """, (track_id,))
    
    # Delete meetings
    conn.execute("""
        DELETE FROM RaceMeetings WHERE TrackID = ?
    """, (track_id,))
    
    # Delete track
    conn.execute("DELETE FROM Tracks WHERE TrackID = ?", (track_id,))
    
    print(f"{track}: Deleted {entry_count:,} entries")

conn.commit()

# Verify
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries")
print(f"Total entries remaining: {cursor.fetchone()[0]:,}")

cursor = conn.execute("SELECT COUNT(*) FROM Tracks")
print(f"Total tracks remaining: {cursor.fetchone()[0]}")

conn.close()
print("\nDone!")
