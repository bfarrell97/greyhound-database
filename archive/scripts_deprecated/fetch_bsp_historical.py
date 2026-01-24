"""
Full BSP Import from Betfair Historical Data API
With track name mapping and improved name normalization
"""
import requests
import json
import sqlite3
import bz2
import re
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

# Map DB track names to BSP venue names
TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

def normalize_name(name):
    """Normalize dog name - remove special chars, uppercase"""
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-]", "", name)  # Remove ' and -
    name = re.sub(r"\s+", " ", name)   # Normalize spaces
    return name.strip()

def download_list_of_files(from_date, to_date):
    url = BASE_URL + 'DownloadListOfFiles'
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": from_date.day,
        "fromMonth": from_date.month,
        "fromYear": from_date.year,
        "toDay": to_date.day,
        "toMonth": to_date.month,
        "toYear": to_date.year,
        "eventId": None,
        "eventName": None,
        "marketTypesCollection": ["WIN"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 200:
        try:
            return resp.json()
        except:
            return []
    return []

def download_file(file_path):
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    headers = {'ssoid': SSOID}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        return resp.content if resp.status_code == 200 else None
    except:
        return None

def process_file(file_path, lookup):
    """Process a single BSP file and return list of (bsp, entry_id) updates"""
    updates = []
    content = download_file(file_path)
    if not content:
        return updates
    
    try:
        decompressed = bz2.decompress(content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        if not lines:
            return updates
        
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return updates
        if 'marketDefinition' not in data['mc'][0]:
            return updates
        
        md = data['mc'][0]['marketDefinition']
        venue = md.get('venue', '').upper()
        market_time = md.get('marketTime', '')[:10]  # YYYY-MM-DD
        
        for r in md.get('runners', []):
            bsp = r.get('bsp')
            if bsp and bsp > 0:
                dog_name = normalize_name(r.get('name', ''))
                key = (venue, market_time, dog_name)
                if key in lookup:
                    updates.append((bsp, lookup[key]))
    except:
        pass
    
    return updates

def main():
    print("="*60, flush=True)
    print("BETFAIR BSP IMPORT - WITH TRACK MAPPING", flush=True)
    print("="*60, flush=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Current BSP coverage
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    initial_bsp = cursor.fetchone()[0]
    print(f"Current BSP coverage: {initial_bsp:,}", flush=True)
    
    # Load all entries needing BSP with normalized names AND mapped track names
    print("\nLoading entries needing BSP...", flush=True)
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP IS NULL
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        # Map track name to BSP venue name
        bsp_track = TRACK_MAPPING.get(track, track)
        normalized_name = normalize_name(dog_name)
        key = (bsp_track, date, normalized_name)
        lookup[key] = entry_id
    
    print(f"Loaded {len(lookup):,} entries needing BSP", flush=True)
    
    # Process month by month (2020 to present)
    all_updates = []
    
    # Start with recent data first (2025 going back)
    months_to_process = []
    for year in [2025, 2024, 2023, 2022, 2021, 2020]:
        for month in range(12, 0, -1):
            if year == 2025 and month > 12:
                continue
            months_to_process.append((year, month))
    
    for year, month in months_to_process:
        # Determine last day of month
        if month == 12:
            last_day = 31
        elif month in [4, 6, 9, 11]:
            last_day = 30
        elif month == 2:
            last_day = 29 if year % 4 == 0 else 28
        else:
            last_day = 31
        
        from_date = datetime(year, month, 1)
        to_date = datetime(year, month, last_day)
        
        print(f"\n{year}-{month:02d}...", flush=True)
        
        files = download_list_of_files(from_date, to_date)
        if not files:
            print(f"  No files or error", flush=True)
            continue
        
        print(f"  {len(files)} files", flush=True)
        
        # Process files in parallel
        month_updates = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_file, fp, lookup): fp for fp in files}
            for future in as_completed(futures):
                updates = future.result()
                month_updates.extend(updates)
                processed += 1
                
                if processed % 500 == 0:
                    print(f"    {processed}/{len(files)} files, {len(month_updates)} matches", flush=True)
        
        print(f"  Matches: {len(month_updates)}", flush=True)
        all_updates.extend(month_updates)
        
        # Commit after each month!
        if month_updates:
            conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", month_updates)
            conn.commit()
            print(f"  Committed to DB", flush=True)
    
    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    
    print("\n" + "="*60, flush=True)
    print(f"COMPLETE!", flush=True)
    print(f"Total BSP values imported: {len(all_updates):,}", flush=True)
    print(f"Final BSP coverage: {final_bsp:,} (+{final_bsp - initial_bsp:,})", flush=True)
    print("="*60, flush=True)
    
    conn.close()

if __name__ == "__main__":
    main()
