"""
Import 5-minute-before prices to database as Price5Min column
Uses local BSP files to extract the price ~5 minutes before race start
"""
import sqlite3
import bz2
import json
import os
import re
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

# Track name mapping (DB -> BSP)
TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def extract_price_at_time(lines, target_seconds=300):
    """Extract price at approximately target_seconds before race start"""
    try:
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return {}
        
        mc = data['mc'][0]
        if 'marketDefinition' not in mc:
            return {}
        
        md = mc['marketDefinition']
        if md.get('marketType') != 'WIN':
            return {}
        
        market_time_str = md.get('marketTime', '')
        try:
            market_time = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
        except:
            return {}
        
        venue = md.get('venue', '').upper()
        market_date = market_time_str[:10]
        
        # Get runner IDs and names
        runner_names = {}
        for r in md.get('runners', []):
            runner_names[r.get('id')] = normalize_name(r.get('name', ''))
        
        # Find prices closest to target time
        runner_prices = {sid: None for sid in runner_names}
        runner_best_diff = {sid: float('inf') for sid in runner_names}
        
        for line in lines[:-1]:
            try:
                d = json.loads(line)
                pt = d.get('pt')
                if not pt:
                    continue
                
                pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                secs = (market_time - pub_time).total_seconds()
                diff = abs(secs - target_seconds)
                
                if diff < 60:  # Within 1 minute of target
                    if 'mc' in d and d['mc']:
                        for r in d['mc'][0].get('rc', []):
                            sid = r.get('id')
                            ltp = r.get('ltp')
                            if sid and ltp and sid in runner_names and diff < runner_best_diff[sid]:
                                runner_best_diff[sid] = diff
                                runner_prices[sid] = ltp
            except:
                continue
        
        # Return dict: (venue, date, name) -> price
        result = {}
        for sid, price in runner_prices.items():
            if price and price > 0:
                name = runner_names.get(sid)
                if name:
                    result[(venue, market_date, name)] = price
        
        return result
    except:
        return {}

def main():
    print("="*60)
    print("IMPORT 5-MIN PRICES TO DATABASE")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Add Price5Min column if not exists
    try:
        conn.execute("ALTER TABLE GreyhoundEntries ADD COLUMN Price5Min REAL")
        conn.commit()
        print("Added Price5Min column")
    except:
        print("Price5Min column already exists")
    
    # Load entries that need Price5Min
    print("\nLoading entries needing Price5Min...")
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Price5Min IS NULL
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        bsp_track = TRACK_MAPPING.get(track, track)
        normalized = normalize_name(dog_name)
        lookup[(bsp_track, date, normalized)] = entry_id
    
    print(f"Loaded {len(lookup):,} entries needing Price5Min")
    
    # Process BSP files
    all_updates = []
    files_done = 0
    
    bsp_folders = [
        r'data\bsp\1\BASIC\2025',
        r'data\bsp\2\BASIC\2024',
        r'data\bsp\2\BASIC\2025',
    ]
    
    for base in bsp_folders:
        if not os.path.exists(base):
            continue
        
        print(f"\n{base}:")
        
        for month in os.listdir(base):
            month_path = os.path.join(base, month)
            if not os.path.isdir(month_path):
                continue
            
            month_updates = 0
            
            for day in os.listdir(month_path):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path):
                    continue
                
                for event in os.listdir(day_path):
                    event_path = os.path.join(day_path, event)
                    if not os.path.isdir(event_path):
                        continue
                    
                    for f in os.listdir(event_path):
                        if not f.endswith('.bz2'):
                            continue
                        
                        files_done += 1
                        
                        try:
                            with bz2.open(os.path.join(event_path, f), 'rt') as fh:
                                lines = fh.readlines()
                            
                            prices = extract_price_at_time(lines, target_seconds=300)
                            
                            for key, price in prices.items():
                                if key in lookup:
                                    all_updates.append((price, lookup[key]))
                                    month_updates += 1
                        except:
                            continue
            
            print(f"  {month}: {month_updates} prices found")
    
    # Update database
    print(f"\nUpdating database with {len(all_updates):,} prices...")
    conn.executemany("UPDATE GreyhoundEntries SET Price5Min = ? WHERE EntryID = ?", all_updates)
    conn.commit()
    
    # Stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    with_price = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries")
    total = cursor.fetchone()[0]
    
    print(f"\nPrice5Min coverage: {with_price:,} / {total:,} ({with_price/total*100:.1f}%)")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
