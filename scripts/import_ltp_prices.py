"""
LTP Price Import - WIN Markets Only
====================================
Imports LTP (Last Traded Price) data from WIN markets at various time intervals.
Basic Plan only includes LTP, not order book depth (ATL/ATB).
PLACE markets don't have LTP in Basic Plan, so they are skipped.
"""
import sqlite3
import requests
import bz2
import json
import re
import time
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SSOID = 'Go3cWrYE2AaXgxNdU0Zs/HMWFmdAV9FAiPnI3wLJI6k='
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

# Time offsets in seconds before market start
TIME_OFFSETS = {
    '1Min': 60,
    '2Min': 120,
    '5Min': 300,
    '10Min': 600,
    '15Min': 900,
    '30Min': 1800,
    '60Min': 3600,
    '2Hr': 7200,
    '3Hr': 10800,
    '6Hr': 21600,
    '12Hr': 43200,
}

# LTP (Last Traded Price) column - single column for now
LTP_COLUMN = 'LTP'

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def download_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return None

def process_win_file(file_path, lookup):
    """Process WIN market file for LTP (Last Traded Price)"""
    updates = []  # List of (ltp, entry_id) tuples
    
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    content = download_with_retry(url, {'ssoid': SSOID})
    
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
        if md.get('marketType') != 'WIN':
            return updates
        
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        # Get runner names and final LTP from last line
        runner_names = {}
        runner_ltp = {}
        for r in md.get('runners', []):
            sid = r.get('id')
            name = normalize_name(r.get('name', ''))
            runner_names[sid] = name
        
        # Get final LTP from last data line (right before market close)
        for line in reversed(lines[:-1]):
            try:
                d = json.loads(line)
                if 'mc' in d and d['mc']:
                    for rc in d['mc'][0].get('rc', []):
                        sid = rc.get('id')
                        ltp = rc.get('ltp')
                        if sid in runner_names and ltp and sid not in runner_ltp:
                            runner_ltp[sid] = ltp
                    # Once we have all runners, break
                    if len(runner_ltp) == len(runner_names):
                        break
            except:
                continue
        
        # Build updates
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            if key not in lookup:
                continue
            entry_id = lookup[key]
            
            if sid in runner_ltp:
                updates.append((runner_ltp[sid], entry_id))
                
    except Exception as e:
        pass
    
    return updates

def get_files_for_month(year, month, market_type='WIN'):
    """Get list of files for a month"""
    last_day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4,6,9,11] else (29 if year%4==0 else 28))
    
    url = BASE_URL + 'DownloadListOfFiles'
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": 1, "fromMonth": month, "fromYear": year,
        "toDay": last_day, "toMonth": month, "toYear": year,
        "eventId": None, "eventName": None,
        "marketTypesCollection": [market_type],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"\n[ERROR] API exception: {e}")
    return []

def add_ltp_column_if_missing(conn):
    """Add LTP column to database if it doesn't exist"""
    cursor = conn.execute("PRAGMA table_info(GreyhoundEntries)")
    existing = {row[1] for row in cursor.fetchall()}
    
    if LTP_COLUMN not in existing:
        conn.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {LTP_COLUMN} REAL")
        conn.commit()
        print(f"Added {LTP_COLUMN} column")
    else:
        print(f"{LTP_COLUMN} column already exists.")

def main():
    print("="*70)
    print("LTP PRICE IMPORT (WIN Markets Only)")
    print("="*70)
    print("Note: Basic Plan only has LTP for WIN markets, not PLACE markets.")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Add LTP column if missing
    add_ltp_column_if_missing(conn)
    
    # Build lookup
    print("\nBuilding lookup...")
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < '2025-12-01'
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        bsp_track = TRACK_MAPPING.get(track, track)
        normalized = normalize_name(dog_name)
        lookup[(bsp_track, date, normalized)] = entry_id
    
    print(f"Entries in lookup: {len(lookup):,}")
    
    # Process months from Nov 2025 backwards to Jan 2020
    months = []
    for year in [2025, 2024, 2023, 2022, 2021, 2020]:
        for month in range(12, 0, -1):
            if year == 2025 and month > 11:
                continue
            months.append((year, month))
    
    total_updated = 0
    
    for year, month in months:
        win_files = get_files_for_month(year, month, 'WIN')
        if not win_files:
            continue
            
        print(f"\n{year}-{month:02d}: {len(win_files)} files", end="", flush=True)
        
        all_updates = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_win_file, fp, lookup): fp for fp in win_files}
            
            for future in as_completed(futures):
                updates = future.result()
                all_updates.extend(updates)
                processed += 1
                if processed % 500 == 0:
                    print(".", end="", flush=True)
        
        # Apply updates
        if all_updates:
            conn.executemany(
                f"UPDATE GreyhoundEntries SET {LTP_COLUMN} = ? WHERE EntryID = ? AND {LTP_COLUMN} IS NULL",
                all_updates
            )
            conn.commit()
        
        print(f" -> {len(all_updates):,} LTP updates")
        total_updated += len(all_updates)
    
    # Final stats
    print("\n" + "="*70)
    print(f"IMPORT COMPLETE! Total LTP updates: {total_updated:,}")
    
    cursor = conn.execute(f"SELECT COUNT(*) FROM GreyhoundEntries WHERE {LTP_COLUMN} IS NOT NULL")
    count = cursor.fetchone()[0]
    print(f"Total {LTP_COLUMN} populated: {count:,}")
    
    print("="*70)
    conn.close()

if __name__ == "__main__":
    main()
