"""
Optimized BSP Import Script
Matches BSP data from Betfair files to GreyhoundEntries by:
- Track (Venue)
- Date (from folder path)
- Greyhound Name (extracted from runner name)
"""
import bz2
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re

DB_PATH = 'greyhound_racing.db'
BSP_FOLDERS = [
    r'data\bsp\1',
    r'data\bsp\2', 
    r'data\bsp\3'
]

def parse_month(month_str):
    """Convert month name to number"""
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    return months.get(month_str, 0)

def extract_dog_name(runner_name):
    """Extract dog name from '1. Dog Name' format"""
    match = re.match(r'\d+\.\s*(.+)', runner_name)
    if match:
        return match.group(1).strip().upper()
    return runner_name.upper()

def process_bsp_file(file_path, date_str, track_mappings):
    """Process a single BSP file and return list of (track, date, dog_name, bsp)"""
    results = []
    
    try:
        with bz2.open(file_path, 'rt') as f:
            lines = f.readlines()
            if not lines:
                return results
            
            # Parse last line for settled BSP
            last_data = json.loads(lines[-1])
            
            if 'mc' not in last_data or not last_data['mc']:
                return results
                
            mc = last_data['mc'][0]
            if 'marketDefinition' not in mc:
                return results
                
            md = mc['marketDefinition']
            venue = md.get('venue', '').upper()
            runners = md.get('runners', [])
            
            # Apply track name mapping
            track = track_mappings.get(venue, venue)
            
            for r in runners:
                bsp = r.get('bsp')
                if bsp is not None and bsp > 0:
                    dog_name = extract_dog_name(r.get('name', ''))
                    results.append((track, date_str, dog_name, bsp))
                    
    except Exception as e:
        pass  # Skip corrupted files silently
        
    return results

def build_track_mappings(conn):
    """Build mapping from Betfair venue names to DB track names"""
    cursor = conn.execute("SELECT DISTINCT TrackName FROM Tracks")
    db_tracks = {row[0].upper(): row[0] for row in cursor.fetchall()}
    
    # Common mappings (add more as needed)
    mappings = {}
    for track in db_tracks:
        mappings[track] = db_tracks[track]
        # Handle variations
        mappings[track.replace(' ', '')] = db_tracks[track]
        
    return mappings

def main():
    print("="*60)
    print("OPTIMIZED BSP IMPORT")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure BSP column exists
    try:
        conn.execute("ALTER TABLE GreyhoundEntries ADD COLUMN BSP REAL")
        conn.commit()
        print("Added BSP column")
    except:
        pass
    
    # Get current BSP coverage
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    current_bsp = cursor.fetchone()[0]
    print(f"Current BSP coverage: {current_bsp:,}")
    
    # Build track mappings
    track_mappings = build_track_mappings(conn)
    print(f"Loaded {len(track_mappings)} track name variations")
    
    # Load all entries into memory for fast lookup
    print("\nLoading GreyhoundEntries for matching...")
    query = """
    SELECT ge.EntryID, t.TrackName, rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP IS NULL
    """
    entries_df = list(conn.execute(query).fetchall())
    print(f"Loaded {len(entries_df):,} entries needing BSP")
    
    # Build lookup dict: (track_upper, date, dog_name_upper) -> entry_id
    lookup = {}
    for entry_id, track, date, dog_name in entries_df:
        key = (track.upper(), date, dog_name.upper())
        lookup[key] = entry_id
    
    print(f"Built lookup with {len(lookup):,} unique keys")
    
    # Process BSP files
    print("\nProcessing BSP files...")
    total_files = 0
    all_updates = []
    
    for folder in BSP_FOLDERS:
        if not os.path.exists(folder):
            continue
            
        for root, dirs, files in os.walk(folder):
            # Extract date from path: folder/BASIC/Year/Month/Day/...
            parts = root.split(os.sep)
            try:
                # Find year/month/day in path
                year = month = day = None
                for i, p in enumerate(parts):
                    if p.isdigit() and len(p) == 4:
                        year = int(p)
                        if i+1 < len(parts):
                            month = parse_month(parts[i+1])
                        if i+2 < len(parts) and parts[i+2].isdigit():
                            day = int(parts[i+2])
                        break
                
                if not (year and month and day):
                    continue
                    
                date_str = f"{year}-{month:02d}-{day:02d}"
                
            except:
                continue
            
            for f in files:
                if f.endswith('.bz2'):
                    total_files += 1
                    file_path = os.path.join(root, f)
                    
                    results = process_bsp_file(file_path, date_str, track_mappings)
                    
                    for track, date, dog_name, bsp in results:
                        key = (track, date, dog_name)
                        if key in lookup:
                            all_updates.append((bsp, lookup[key]))
                    
                    if total_files % 1000 == 0:
                        print(f"  Processed {total_files:,} files, found {len(all_updates):,} matches...", flush=True)
    
    print(f"\nTotal files processed: {total_files:,}")
    print(f"Total BSP matches found: {len(all_updates):,}")
    
    # Batch update
    if all_updates:
        print("\nUpdating database...")
        conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", all_updates)
        conn.commit()
        print(f"Updated {len(all_updates):,} entries with BSP values")
    
    # Final coverage
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    print(f"\nFinal BSP coverage: {final_bsp:,} (+{final_bsp - current_bsp:,} new)")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
