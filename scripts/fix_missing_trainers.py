import sqlite3
import pandas as pd

def fix_trainers():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Finding entries with NULL TrainerID for TODAY (2025-12-26)...")
    
    # Get all NULL trainer entries for today
    query = """
    SELECT ge.EntryID, ge.GreyhoundID
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate = '2025-12-26'
    AND ge.TrainerID IS NULL
    """
    rows = cursor.execute(query).fetchall()
    
    print(f"Found {len(rows)} entries with missing TrainerID.")
    
    fixed_count = 0
    
    for entry_id, dog_id in rows:
        # Find LAST known TrainerID for this dog
        # Order by Date DESC, skip the current NULL one
        t_query = """
        SELECT ge.TrainerID
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.GreyhoundID = ?
        AND ge.TrainerID IS NOT NULL
        ORDER BY rm.MeetingDate DESC
        LIMIT 1
        """
        last_trainer = cursor.execute(t_query, (dog_id,)).fetchone()
        
        if last_trainer:
            tid = last_trainer[0]
            # Update
            cursor.execute("UPDATE GreyhoundEntries SET TrainerID = ? WHERE EntryID = ?", (tid, entry_id))
            fixed_count += 1
            
    conn.commit()
    print(f"Fixed {fixed_count} / {len(rows)} Trainers.")
    conn.close()

if __name__ == "__main__":
    fix_trainers()
