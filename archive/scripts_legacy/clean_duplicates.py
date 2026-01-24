import sqlite3

def clean_duplicates():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Finding duplicate greyhounds...")
    # Find names that appear multiple times (case insensitive)
    cursor.execute("""
        SELECT UPPER(GreyhoundName), COUNT(*)
        FROM Greyhounds
        GROUP BY UPPER(GreyhoundName)
        HAVING COUNT(*) > 1
    """)
    duplicates = cursor.fetchall()
    print(f"Found {len(duplicates)} duplicate names.")
    
    for name, count in duplicates:
        print(f"Processing: {name}")
        # Get all IDs for this name
        cursor.execute("SELECT GreyhoundID, GreyhoundName FROM Greyhounds WHERE UPPER(GreyhoundName) = ?", (name,))
        rows = cursor.fetchall()
        
        # Strategy: Keep the one with the lowest ID (usually original fasttrack with history)
        # UNLESS the newer one is the only one with Odds?
        # Actually, best to Keep Lowest ID.
        ids = [r[0] for r in rows]
        ids.sort()
        keep_id = ids[0]
        remove_ids = ids[1:]
        
        print(f"  Keep: {keep_id}, Remove: {remove_ids}")
        
        for remove_id in remove_ids:
            # Update Entries
            # CAUTION: If we update Entry to point to KeepID, and KeepID ALREADY has an entry in that race, we create a duplicate entry.
            # We must handle this.
            
            # 1. Update Entries where new conflict doesn't exist
            # (Too complex to do in one SQL).
            
            # Simpler approach:
            # For each RaceID where 'remove_id' has an entry:
            #   Check if 'keep_id' already has an entry.
            #   If YES: Update 'keep_id' entry with data from 'remove_id' (if useful), then DELETE 'remove_id' entry.
            #   If NO: Update 'remove_id' entry to point to 'keep_id'.
            
            cursor.execute("SELECT EntryID, RaceID, StartingPrice FROM GreyhoundEntries WHERE GreyhoundID = ?", (remove_id,))
            entries_to_move = cursor.fetchall()
            
            for entry_id, race_id, sp in entries_to_move:
                # Check if keep_id is in this race
                cursor.execute("SELECT EntryID, StartingPrice FROM GreyhoundEntries WHERE RaceID = ? AND GreyhoundID = ?", (race_id, keep_id))
                existing_entry = cursor.fetchone()
                
                if existing_entry:
                    # Conflict! keep_id is already here.
                    existing_entry_id = existing_entry[0]
                    existing_sp = existing_entry[1]
                    
                    # If the one we are removing has odds, and existing doesn't, COPY IT OVER
                    if sp is not None and existing_sp is None:
                        print(f"    Merging Odds {sp} from {entry_id} to {existing_entry_id}")
                        cursor.execute("UPDATE GreyhoundEntries SET StartingPrice = ? WHERE EntryID = ?", (sp, existing_entry_id))
                    
                    # Now delete the redundant entry
                    print(f"    Deleting redundant entry {entry_id}")
                    cursor.execute("DELETE FROM GreyhoundEntries WHERE EntryID = ?", (entry_id,))
                else:
                    # No conflict, just reassign owner
                    print(f"    Reassigning Entry {entry_id} to Dog {keep_id}")
                    cursor.execute("UPDATE GreyhoundEntries SET GreyhoundID = ? WHERE EntryID = ?", (keep_id, entry_id))
            
            # Finally remove the dog
            cursor.execute("DELETE FROM Greyhounds WHERE GreyhoundID = ?", (remove_id,))
            
    conn.commit()
    conn.close()
    print("Cleanup Complete.")

if __name__ == "__main__":
    clean_duplicates()
