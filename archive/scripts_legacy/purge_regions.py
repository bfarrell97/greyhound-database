
import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

def purge_regions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Identify Tracks
    print("Identifying tracks to purge...")
    query = """
    SELECT TrackID, TrackName, State
    FROM Tracks
    WHERE State IN ('TAS', 'Tasmania', 'NZ', 'New Zealand')
       OR TrackName LIKE '%(NZ)%'
       OR TrackName IN ('Hobart', 'Launceston', 'Devonport', 'Dport @ LCN', 'Dport @ HOB')
    """
    
    tracks_df = pd.read_sql_query(query, conn)
    
    if len(tracks_df) == 0:
        print("No tracks found matching criteria.")
        return

    print(f"\nFound {len(tracks_df)} tracks to purge:")
    print(tracks_df[['TrackName', 'State']].to_string())
    
    track_ids = tracks_df['TrackID'].tolist()
    track_ids_str = ",".join(map(str, track_ids))
    
    # 2. Count Impact
    print("\nCounting records to be deleted...")
    
    # Meetings
    cursor.execute(f"SELECT COUNT(*) FROM RaceMeetings WHERE TrackID IN ({track_ids_str})")
    meetings_count = cursor.fetchone()[0]
    
    # Races (need to join)
    cursor.execute(f"""
        SELECT COUNT(*) FROM Races r
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.TrackID IN ({track_ids_str})
    """)
    races_count = cursor.fetchone()[0]
    
    # Entries
    cursor.execute(f"""
        SELECT COUNT(*) FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.TrackID IN ({track_ids_str})
    """)
    entries_count = cursor.fetchone()[0]
    
    print(f"  - Tracks: {len(tracks_df)}")
    print(f"  - Meetings: {meetings_count}")
    print(f"  - Races: {races_count}")
    print(f"  - Entries: {entries_count}")
    
    # Bypass confirmation for automation
    # confirm = input("\nType 'YES' to proceed with deletion: ")
    # if confirm != 'YES':
    #     print("Aborted.")
    #     return

    # 3. Execute Deletion
    print("\nExecuting purge...")
    
    # Delete Entries
    cursor.execute(f"""
        DELETE FROM GreyhoundEntries WHERE RaceID IN (
            SELECT RaceID FROM Races r
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE rm.TrackID IN ({track_ids_str})
        )
    """)
    print(f"Deleted {cursor.rowcount} entries.")
    
    # Delete Races
    cursor.execute(f"""
        DELETE FROM Races WHERE MeetingID IN (
            SELECT MeetingID FROM RaceMeetings
            WHERE TrackID IN ({track_ids_str})
        )
    """)
    print(f"Deleted {cursor.rowcount} races.")
    
    # Delete Meetings
    cursor.execute(f"DELETE FROM RaceMeetings WHERE TrackID IN ({track_ids_str})")
    print(f"Deleted {cursor.rowcount} meetings.")
    
    # Delete Tracks
    cursor.execute(f"DELETE FROM Tracks WHERE TrackID IN ({track_ids_str})")
    print(f"Deleted {cursor.rowcount} tracks.")
    
    conn.commit()
    conn.close()
    print("Purge complete.")

if __name__ == "__main__":
    purge_regions()
