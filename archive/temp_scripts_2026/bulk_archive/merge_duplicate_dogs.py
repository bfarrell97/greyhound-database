import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

def merge_duplicates_safe():
    print("="*60)
    print("MERGING DUPLICATE DOGS (SAFE MODE)")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch all dogs
    print("Fetching all dogs...")
    df = pd.read_sql_query("SELECT GreyhoundID, GreyhoundName FROM Greyhounds", conn)
    
    norm_map = {} # {NORMALIZED_NAME: MASTER_ID}
    merges = []   # [(DUPLICATE_ID, MASTER_ID, Name)]
    
    # Normalize: Trim, Upper, Remove double spaces
    df['NormName'] = df['GreyhoundName'].apply(lambda x: " ".join(str(x).upper().split()))
    
    # Sort: Master should be the FIRST encountered (Lowest ID usually)
    df = df.sort_values('GreyhoundID')
    
    for _, row in df.iterrows():
        norm = row['NormName']
        gid = row['GreyhoundID']
        
        if norm in norm_map:
            master_id = norm_map[norm]
            if master_id != gid: # Ignore self matches
                merges.append((gid, master_id, norm))
        else:
            norm_map[norm] = gid
            
    print(f"Found {len(merges)} duplicates to merge.")
    if not merges:
        conn.close()
        return

    print("Beginning Safe Merge...")
    
    merged_entries = 0
    collided_entries = 0
    
    for dup_id, master_id, name in merges:
        # Get entries for duplicate dog
        cursor.execute("SELECT EntryID, RaceID FROM GreyhoundEntries WHERE GreyhoundID = ?", (dup_id,))
        dup_entries = cursor.fetchall() # List of (EntryID, RaceID)
        
        for entry_id, race_id in dup_entries:
            # Check if Master is already in this race
            cursor.execute("SELECT EntryID FROM GreyhoundEntries WHERE RaceID = ? AND GreyhoundID = ?", (race_id, master_id))
            master_entry = cursor.fetchone()
            
            if master_entry:
                # COLLISION: Master is already here.
                # We assume Duplicate Entry has recent Price data (from today's scrape)?
                # We assume Master Entry has Form data (from history)?
                # Merge logic: Copy NON-NULL values from Duplicate -> Master?
                # Actually, simpliest approach: Just delete Duplicate Entry.
                # Why? Because if Master exists, it probably came from a valid import.
                # But wait, did the Scraper put PRICES on the Duplicate? Yes.
                # So we must copy prices.
                
                # Fetch Duplicate Prices
                cursor.execute("SELECT Price5Min, BSP FROM GreyhoundEntries WHERE EntryID = ?", (entry_id,))
                prices = cursor.fetchone()
                
                if prices:
                    p5, bsp = prices
                    updates = []
                    params = []
                    if p5: 
                        updates.append("Price5Min = ?")
                        params.append(p5)
                    if bsp:
                        updates.append("BSP = ?")
                        params.append(bsp)
                        
                    if updates:
                        params.append(master_entry[0]) # Where EntryID = MasterEntryID
                        sql = f"UPDATE GreyhoundEntries SET {', '.join(updates)} WHERE EntryID = ?"
                        cursor.execute(sql, params)
                
                # Delete Duplicate Entry
                cursor.execute("DELETE FROM GreyhoundEntries WHERE EntryID = ?", (entry_id,))
                collided_entries += 1
                
            else:
                # NO COLLISION: Safe to switch ID
                cursor.execute("UPDATE GreyhoundEntries SET GreyhoundID = ? WHERE EntryID = ?", (master_id, entry_id))
                merged_entries += 1
                
        # Finally delete Duplicate Dog itself
        cursor.execute("DELETE FROM Greyhounds WHERE GreyhoundID = ?", (dup_id,))
        
    conn.commit()
    print(f"Merge Complete.")
    print(f"  Entries Switched: {merged_entries}")
    print(f"  Entries Merged (Collision): {collided_entries}")
    print(f"  Duplicate Dogs Deleted: {len(merges)}")
    conn.close()

if __name__ == "__main__":
    merge_duplicates_safe()
