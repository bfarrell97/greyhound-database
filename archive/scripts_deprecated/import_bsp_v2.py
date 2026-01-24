"""
Simplified BSP Import - with debug output
"""
import bz2
import json
import os
import sqlite3
import re

DB_PATH = 'greyhound_racing.db'

def extract_dog_name(runner_name):
    """Extract dog name from '1. Dog Name' format"""
    match = re.match(r'\d+\.\s*(.+)', runner_name)
    if match:
        return match.group(1).strip().upper()
    return runner_name.upper()

def main():
    print("="*60, flush=True)
    print("BSP IMPORT v2 - Debug Mode", flush=True)
    print("="*60, flush=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Current coverage
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    current_bsp = cursor.fetchone()[0]
    print(f"Current BSP coverage: {current_bsp:,}", flush=True)
    
    # Get DB track names
    db_tracks = {}
    for row in conn.execute("SELECT DISTINCT TrackName FROM Tracks").fetchall():
        db_tracks[row[0].upper()] = row[0]
    print(f"Loaded {len(db_tracks)} tracks from DB", flush=True)
    
    # Load entries needing BSP
    print("\nLoading entries needing BSP...", flush=True)
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, UPPER(g.GreyhoundName)
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP IS NULL
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        lookup[(track, date, dog_name)] = entry_id
    
    print(f"Loaded {len(lookup):,} entries needing BSP", flush=True)
    
    # Process BSP folders
    base_folders = [
        r'data\bsp\1\BASIC',
        r'data\bsp\2\BASIC',
        r'data\bsp\3\BASIC'
    ]
    
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    
    total_files = 0
    all_updates = []
    
    for base in base_folders:
        if not os.path.exists(base):
            print(f"Skipping (not found): {base}", flush=True)
            continue
        
        print(f"\nProcessing: {base}", flush=True)
        
        # Walk: BASIC/Year/Month/Day/EventID/MarketID.bz2
        for year_name in os.listdir(base):
            year_path = os.path.join(base, year_name)
            if not os.path.isdir(year_path) or not year_name.isdigit():
                continue
            
            for month_name in os.listdir(year_path):
                month_path = os.path.join(year_path, month_name)
                if not os.path.isdir(month_path) or month_name not in months:
                    continue
                
                month_num = months[month_name]
                
                for day_name in os.listdir(month_path):
                    day_path = os.path.join(month_path, day_name)
                    if not os.path.isdir(day_path) or not day_name.isdigit():
                        continue
                    
                    date_str = f"{year_name}-{month_num}-{int(day_name):02d}"
                    
                    # Event folders
                    for event_id in os.listdir(day_path):
                        event_path = os.path.join(day_path, event_id)
                        if not os.path.isdir(event_path):
                            continue
                        
                        # Market files
                        for market_file in os.listdir(event_path):
                            if not market_file.endswith('.bz2'):
                                continue
                            
                            total_files += 1
                            file_path = os.path.join(event_path, market_file)
                            
                            try:
                                with bz2.open(file_path, 'rt') as f:
                                    lines = f.readlines()
                                    if not lines:
                                        continue
                                    
                                    # Get last line with BSP data
                                    last_data = json.loads(lines[-1])
                                    if 'mc' not in last_data or not last_data['mc']:
                                        continue
                                    
                                    mc = last_data['mc'][0]
                                    if 'marketDefinition' not in mc:
                                        continue
                                    
                                    md = mc['marketDefinition']
                                    venue = md.get('venue', '').upper()
                                    runners = md.get('runners', [])
                                    
                                    for r in runners:
                                        bsp = r.get('bsp')
                                        if bsp and bsp > 0:
                                            dog_name = extract_dog_name(r.get('name', ''))
                                            key = (venue, date_str, dog_name)
                                            if key in lookup:
                                                all_updates.append((bsp, lookup[key]))
                            except:
                                pass
                            
                            if total_files % 1000 == 0:
                                print(f"  Files: {total_files:,}, Matches: {len(all_updates):,}", flush=True)
    
    print(f"\nTotal files: {total_files:,}", flush=True)
    print(f"Total matches: {len(all_updates):,}", flush=True)
    
    if all_updates:
        print("\nUpdating database...", flush=True)
        conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", all_updates)
        conn.commit()
        print(f"Updated {len(all_updates):,} entries", flush=True)
    
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    print(f"\nFinal BSP coverage: {final_bsp:,} (+{final_bsp - current_bsp:,})", flush=True)
    
    conn.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
