"""
Lay & PlaceLay Price Import - 5 Threads
========================================
Imports Lay prices from WIN markets and PlaceLay prices from PLACE markets.
Adds new columns to database if they don't exist.
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

# Columns to add for LAY (WIN market)
LAY_COLUMNS = [
    'LayBSP', 'LayLTP', 'LayOpen',
    'Lay1Min', 'Lay2Min', 'Lay5Min', 'Lay10Min', 'Lay15Min',
    'Lay30Min', 'Lay60Min', 'Lay2Hr', 'Lay3Hr', 'Lay6Hr', 'Lay12Hr'
]

# Columns to add for PLACELAY (PLACE market)
PLACELAY_COLUMNS = [
    'PlaceLayBSP', 'PlaceOpen', 'PlaceLayOpen',
    'PlaceLay1Min', 'PlaceLay2Min', 'PlaceLay5Min', 'PlaceLay10Min',
    'PlaceLay15Min', 'PlaceLay30Min', 'PlaceLay60Min'
]

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

def get_best_lay_price(runner_data):
    """Extract best available lay price from runner data"""
    atl = runner_data.get('atl', [])
    if atl and len(atl) > 0:
        # atl is list of [price, size] pairs, sorted by price ascending
        # Best lay = lowest price available to lay
        if isinstance(atl[0], list):
            return atl[0][0]  # First element, first value = price
        else:
            return atl[0]  # Sometimes just the price
    return None

# Debug counters (global for aggregation)
import threading
_debug_lock = threading.Lock()
_debug_stats = {'files': 0, 'matches': 0, 'misses': 0, 'sample_misses': []}

def process_win_file(file_path, lookup):
    """Process WIN market file for Lay prices"""
    global _debug_stats
    updates = {col: [] for col in LAY_COLUMNS}
    
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
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return updates
        
        # Get runner info and BSP lay
        runner_names = {}
        runner_lay_bsp = {}
        for r in md.get('runners', []):
            sid = r.get('id')
            name = normalize_name(r.get('name', ''))
            runner_names[sid] = name
            # Lay BSP from marketDefinition
            bsp = r.get('bsp')
            if bsp and bsp > 0:
                runner_lay_bsp[sid] = bsp  # BSP is same for back/lay

        # Track best lay prices at each time offset
        runner_lay_prices = {sid: {} for sid in runner_names}
        runner_best_diff = {sid: {k: float('inf') for k in TIME_OFFSETS} for sid in runner_names}
        runner_ltp = {sid: None for sid in runner_names}
        runner_open = {sid: None for sid in runner_names}
        first_timestamp = None
        
        for line in lines[:-1]:
            try:
                d = json.loads(line)
                pt = d.get('pt')
                if not pt:
                    continue
                pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                secs = (market_time - pub_time).total_seconds()
                
                if first_timestamp is None:
                    first_timestamp = pt
                
                if 'mc' in d and d['mc']:
                    for r in d['mc'][0].get('rc', []):
                        sid = r.get('id')
                        if sid not in runner_names:
                            continue
                        
                        lay_price = get_best_lay_price(r)
                        if not lay_price:
                            continue
                        
                        # Track first available price as "Open"
                        if pt == first_timestamp and runner_open[sid] is None:
                            runner_open[sid] = lay_price
                        
                        # Track LTP for lay (last seen)
                        runner_ltp[sid] = lay_price
                        
                        # Match to time offsets
                        for label, target_secs in TIME_OFFSETS.items():
                            diff = abs(secs - target_secs)
                            if diff < 60 and diff < runner_best_diff[sid][label]:
                                runner_best_diff[sid][label] = diff
                                runner_lay_prices[sid][label] = lay_price
            except:
                continue
        
        # Build updates
        local_matches = 0
        local_misses = 0
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            if key not in lookup:
                local_misses += 1
                with _debug_lock:
                    if len(_debug_stats['sample_misses']) < 10:
                        _debug_stats['sample_misses'].append(key)
                continue
            local_matches += 1
            with _debug_lock:
                _debug_stats['matches'] += 1
            entry_id = lookup[key]
            
            # LayBSP
            if sid in runner_lay_bsp:
                updates['LayBSP'].append((runner_lay_bsp[sid], entry_id))
            
            # LayLTP
            if runner_ltp.get(sid):
                updates['LayLTP'].append((runner_ltp[sid], entry_id))
            
            # LayOpen
            if runner_open.get(sid):
                updates['LayOpen'].append((runner_open[sid], entry_id))
            
            # Time-based columns
            for label in TIME_OFFSETS:
                col = f'Lay{label}'
                if runner_lay_prices[sid].get(label):
                    updates[col].append((runner_lay_prices[sid][label], entry_id))
    except:
        pass
    
    return updates

def process_place_file(file_path, lookup):
    """Process PLACE market file for PlaceLay prices"""
    updates = {col: [] for col in PLACELAY_COLUMNS}
    
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
        if md.get('marketType') != 'PLACE':
            return updates
        
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return updates
        
        runner_names = {}
        runner_lay_bsp = {}
        for r in md.get('runners', []):
            sid = r.get('id')
            name = normalize_name(r.get('name', ''))
            runner_names[sid] = name
            bsp = r.get('bsp')
            if bsp and bsp > 0:
                runner_lay_bsp[sid] = bsp

        # Subset of time offsets for place markets
        place_offsets = {k: v for k, v in TIME_OFFSETS.items() if v <= 3600}  # Up to 60min
        
        runner_lay_prices = {sid: {} for sid in runner_names}
        runner_best_diff = {sid: {k: float('inf') for k in place_offsets} for sid in runner_names}
        runner_back_open = {sid: None for sid in runner_names}
        runner_lay_open = {sid: None for sid in runner_names}
        first_timestamp = None
        
        for line in lines[:-1]:
            try:
                d = json.loads(line)
                pt = d.get('pt')
                if not pt:
                    continue
                pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                secs = (market_time - pub_time).total_seconds()
                
                if first_timestamp is None:
                    first_timestamp = pt
                
                if 'mc' in d and d['mc']:
                    for r in d['mc'][0].get('rc', []):
                        sid = r.get('id')
                        if sid not in runner_names:
                            continue
                        
                        # Track Open prices (first available)
                        if pt == first_timestamp:
                            ltp = r.get('ltp')
                            if ltp and runner_back_open[sid] is None:
                                runner_back_open[sid] = ltp
                            lay_price = get_best_lay_price(r)
                            if lay_price and runner_lay_open[sid] is None:
                                runner_lay_open[sid] = lay_price
                        
                        lay_price = get_best_lay_price(r)
                        if not lay_price:
                            continue
                        
                        for label, target_secs in place_offsets.items():
                            diff = abs(secs - target_secs)
                            if diff < 60 and diff < runner_best_diff[sid][label]:
                                runner_best_diff[sid][label] = diff
                                runner_lay_prices[sid][label] = lay_price
            except:
                continue
        
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            if key not in lookup:
                continue
            entry_id = lookup[key]
            
            if sid in runner_lay_bsp:
                updates['PlaceLayBSP'].append((runner_lay_bsp[sid], entry_id))
            
            # PlaceOpen (back)
            if runner_back_open.get(sid):
                updates['PlaceOpen'].append((runner_back_open[sid], entry_id))
            
            # PlaceLayOpen
            if runner_lay_open.get(sid):
                updates['PlaceLayOpen'].append((runner_lay_open[sid], entry_id))
            
            for label in place_offsets:
                col = f'PlaceLay{label}'
                if col in updates and runner_lay_prices[sid].get(label):
                    updates[col].append((runner_lay_prices[sid][label], entry_id))
    except:
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
            result = resp.json()
            if not result:
                print(f"\n[DEBUG] {year}-{month:02d} {market_type}: Empty response")
            return result
        else:
            print(f"\n[ERROR] API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"\n[ERROR] API exception: {e}")
    return []

def add_columns_if_missing(conn):
    """Add new columns to database if they don't exist"""
    cursor = conn.execute("PRAGMA table_info(GreyhoundEntries)")
    existing = {row[1] for row in cursor.fetchall()}
    
    all_new_cols = LAY_COLUMNS + PLACELAY_COLUMNS
    added = []
    
    for col in all_new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {col} REAL")
            added.append(col)
    
    if added:
        conn.commit()
        print(f"Added {len(added)} new columns: {', '.join(added)}")
    else:
        print("All columns already exist.")

def main():
    print("="*70)
    print("LAY & PLACELAY PRICE IMPORT (5 Threads)")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Add columns if missing
    add_columns_if_missing(conn)
    
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
            if year == 2025 and month > 11:  # Start from Nov 2025
                continue
            if year == 2020 and month < 1:  # End at Jan 2020
                continue
            months.append((year, month))
    
    for year, month in months:
        # Process WIN markets for Lay prices
        win_files = get_files_for_month(year, month, 'WIN')
        if win_files:
            print(f"\n{year}-{month:02d} WIN: {len(win_files)} files", end="", flush=True)
            
            all_updates = {col: [] for col in LAY_COLUMNS}
            processed = 0
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_win_file, fp, lookup): fp for fp in win_files}
                
                for future in as_completed(futures):
                    updates = future.result()
                    for col, vals in updates.items():
                        all_updates[col].extend(vals)
                    processed += 1
                    if processed % 500 == 0:
                        print(".", end="", flush=True)
            
            # Apply updates
            for col, vals in all_updates.items():
                if vals:
                    conn.executemany(
                        f"UPDATE GreyhoundEntries SET {col} = ? WHERE EntryID = ? AND {col} IS NULL",
                        vals
                    )
            conn.commit()
            
            total = sum(len(v) for v in all_updates.values())
            print(f" -> {total:,} updates")
            
            # Debug output
            if _debug_stats['sample_misses']:
                print(f"[DEBUG] Sample misses (Venue, Date, Dog):")
                for miss in _debug_stats['sample_misses'][:5]:
                    print(f"        {miss}")
                _debug_stats['sample_misses'] = []  # Reset for next month
        
        # Process PLACE markets for PlaceLay prices
        place_files = get_files_for_month(year, month, 'PLACE')
        if place_files:
            print(f"{year}-{month:02d} PLACE: {len(place_files)} files", end="", flush=True)
            
            all_updates = {col: [] for col in PLACELAY_COLUMNS}
            processed = 0
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_place_file, fp, lookup): fp for fp in place_files}
                
                for future in as_completed(futures):
                    updates = future.result()
                    for col, vals in updates.items():
                        all_updates[col].extend(vals)
                    processed += 1
                    if processed % 500 == 0:
                        print(".", end="", flush=True)
            
            for col, vals in all_updates.items():
                if vals:
                    conn.executemany(
                        f"UPDATE GreyhoundEntries SET {col} = ? WHERE EntryID = ? AND {col} IS NULL",
                        vals
                    )
            conn.commit()
            
            total = sum(len(v) for v in all_updates.values())
            print(f" -> {total:,} updates")
    
    # Final stats
    print("\n" + "="*70)
    print("IMPORT COMPLETE!")
    
    for col in LAY_COLUMNS[:3]:  # Sample columns
        cursor = conn.execute(f"SELECT COUNT(*) FROM GreyhoundEntries WHERE {col} IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"{col}: {count:,}")
    
    for col in PLACELAY_COLUMNS[:3]:
        cursor = conn.execute(f"SELECT COUNT(*) FROM GreyhoundEntries WHERE {col} IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"{col}: {count:,}")
    
    print("="*70)
    conn.close()

if __name__ == "__main__":
    main()
