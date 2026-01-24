"""
BSP Import v3 - Month-by-month with progress
"""
import bz2, json, os, sqlite3, re, sys
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def extract_dog_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    return match.group(1).strip().upper() if match else name.upper()

def main():
    print("="*60, flush=True)
    print("BSP IMPORT v3 - Month by Month", flush=True)
    print("="*60, flush=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Current coverage
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    current_bsp = cursor.fetchone()[0]
    print(f"Current BSP: {current_bsp:,}", flush=True)
    
    # Load entries
    print("Loading entries...", flush=True)
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
    print(f"Lookup: {len(lookup):,} entries", flush=True)
    
    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    
    base_folders = [r'data\bsp\1\BASIC', r'data\bsp\2\BASIC', r'data\bsp\3\BASIC']
    
    total_files = 0
    total_matches = 0
    all_updates = []
    
    for base in base_folders:
        if not os.path.exists(base):
            continue
        
        print(f"\n{base}:", flush=True)
        
        for year_name in sorted(os.listdir(base)):
            year_path = os.path.join(base, year_name)
            if not os.path.isdir(year_path) or not year_name.isdigit():
                continue
            
            for month_name in months.keys():
                month_path = os.path.join(year_path, month_name)
                if not os.path.exists(month_path):
                    continue
                
                month_num = months[month_name]
                month_files = 0
                month_matches = 0
                
                # Process each day
                for day_name in os.listdir(month_path):
                    day_path = os.path.join(month_path, day_name)
                    if not os.path.isdir(day_path) or not day_name.isdigit():
                        continue
                    
                    date_str = f"{year_name}-{month_num}-{int(day_name):02d}"
                    
                    for event_id in os.listdir(day_path):
                        event_path = os.path.join(day_path, event_id)
                        if not os.path.isdir(event_path):
                            continue
                        
                        for market_file in os.listdir(event_path):
                            if not market_file.endswith('.bz2'):
                                continue
                            
                            month_files += 1
                            file_path = os.path.join(event_path, market_file)
                            
                            try:
                                with bz2.open(file_path, 'rt') as f:
                                    lines = f.readlines()
                                    if not lines:
                                        continue
                                    
                                    data = json.loads(lines[-1])
                                    if 'mc' not in data or not data['mc']:
                                        continue
                                    mc = data['mc'][0]
                                    if 'marketDefinition' not in mc:
                                        continue
                                    
                                    md = mc['marketDefinition']
                                    venue = md.get('venue', '').upper()
                                    
                                    for r in md.get('runners', []):
                                        bsp = r.get('bsp')
                                        if bsp and bsp > 0:
                                            dog_name = extract_dog_name(r.get('name', ''))
                                            key = (venue, date_str, dog_name)
                                            if key in lookup:
                                                all_updates.append((bsp, lookup[key]))
                                                month_matches += 1
                            except:
                                pass
                
                total_files += month_files
                total_matches += month_matches
                print(f"  {year_name}-{month_name}: {month_files:,} files, {month_matches:,} matches", flush=True)
    
    print(f"\nTotal: {total_files:,} files, {total_matches:,} matches", flush=True)
    
    if all_updates:
        print(f"Updating {len(all_updates):,} entries...", flush=True)
        conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", all_updates)
        conn.commit()
    
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    print(f"\nFinal BSP: {final_bsp:,} (+{final_bsp - current_bsp:,})", flush=True)
    
    conn.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
