"""
Import ALL Price Timestamps from Betfair Historical Data
========================================================
Extracts LTP at multiple time points before race:
- Price1Min, Price2Min, Price5Min, Price10Min, Price15Min, Price30Min, Price60Min
Uses 5 threads for faster processing.
"""
import sqlite3
import requests
import bz2
import json
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_URL = "https://historicdata.betfair.com/api/"
SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0="

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

# Time points to extract (seconds before race start)
TIME_POINTS = {
    'Price1Min': 60,
    'Price2Min': 120,
    'Price5Min': 300,
    'Price10Min': 600,
    'Price15Min': 900,
    'Price30Min': 1800,
    'Price60Min': 3600,
    'Price2Hr': 7200,
    'Price3Hr': 10800,
    'Price6Hr': 21600,
    'Price12Hr': 43200,
}

# Also track opening/earliest price
TRACK_OPENING_PRICE = True

import re

def normalize_name(name):
    if not name:
        return ''
    
    # Logic from import_5threads.py
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    
    name = name.upper()
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    
    for old, new in TRACK_MAPPING.items():
        name = name.replace(old, new)
        
    return name

def download_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.content
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None

def process_file(file_path, lookup):
    """Process one file and return price updates for all time points"""
    updates = {tp: [] for tp in TIME_POINTS}
    bsp_updates = []
    
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    content = download_with_retry(url, {'ssoid': SSOID})
    
    if not content:
        return bsp_updates, updates
    
    try:
        decompressed = bz2.decompress(content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        if not lines:
            return bsp_updates, updates
        
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return bsp_updates, updates
        if 'marketDefinition' not in data['mc'][0]:
            return bsp_updates, updates
        
        md = data['mc'][0]['marketDefinition']
        if md.get('marketType') != 'WIN':
            return bsp_updates, updates
        
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return bsp_updates, updates
        
        # Get BSP and runner names
        runner_bsp = {}
        runner_names = {}
        for r in md.get('runners', []):
            sid = r.get('id')
            bsp = r.get('bsp')
            name = normalize_name(r.get('name', ''))
            if bsp and bsp > 0:
                runner_bsp[sid] = bsp
                runner_names[sid] = name
        
        # Initialize price tracking for each time point
        runner_prices = {tp: {sid: None for sid in runner_names} for tp in TIME_POINTS}
        runner_best_diff = {tp: {sid: float('inf') for sid in runner_names} for tp in TIME_POINTS}
        
        # Parse all lines to find prices at each time point
        for line in lines[:-1]:
            try:
                d = json.loads(line)
                pt = d.get('pt')
                if not pt:
                    continue
                pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                secs_before = (market_time - pub_time).total_seconds()
                
                # Check each time point
                for tp_name, target_secs in TIME_POINTS.items():
                    diff = abs(secs_before - target_secs)
                    
                    # Accept prices within tolerance of target
                    # More tolerance for earlier prices (less trading activity)
                    if target_secs <= 300:
                        tolerance = 30
                    elif target_secs <= 3600:
                        tolerance = 60
                    else:
                        tolerance = 300  # 5 min tolerance for hour+ prices
                    
                    if diff < tolerance:
                        if 'mc' in d and d['mc']:
                            for r in d['mc'][0].get('rc', []):
                                sid = r.get('id')
                                ltp = r.get('ltp')
                                if sid and ltp and sid in runner_names and diff < runner_best_diff[tp_name][sid]:
                                    runner_best_diff[tp_name][sid] = diff
                                    runner_prices[tp_name][sid] = ltp
            except:
                continue
        
        # Create updates
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            if key in lookup:
                entry_id = lookup[key]
                
                # BSP update
                bsp = runner_bsp.get(sid)
                if bsp:
                    bsp_updates.append((bsp, entry_id))
                
                # Price updates for each time point
                for tp_name in TIME_POINTS:
                    price = runner_prices[tp_name].get(sid)
                    if price and price > 0:
                        updates[tp_name].append((price, entry_id))
    except:
        pass
    
    return bsp_updates, updates

def get_files_for_month(year, month):
    """Get list of files for a month"""
    last_day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4,6,9,11] else (29 if year%4==0 else 28))
    
    url = BASE_URL + 'DownloadListOfFiles'
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": 1, "fromMonth": month, "fromYear": year,
        "toDay": last_day, "toMonth": month, "toYear": year,
        "eventId": None, "eventName": None,
        "marketTypesCollection": ["WIN"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return []

def main():
    print("="*70)
    print("MULTI-TIMESTAMP PRICE IMPORT")
    print("Extracting: " + ", ".join(TIME_POINTS.keys()))
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Ensure all columns exist
    existing_cols = [r[1] for r in cursor.execute('PRAGMA table_info(GreyhoundEntries)').fetchall()]
    for col in TIME_POINTS:
        if col not in existing_cols:
            cursor.execute(f'ALTER TABLE GreyhoundEntries ADD COLUMN {col} REAL')
    conn.commit()
    
    # Build lookup
    print("Building lookup table...")
    cursor.execute("""
        SELECT ge.EntryID, t.TrackName, rm.MeetingDate, g.GreyhoundName
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    """)
    
    lookup = {}
    for entry_id, track, date, name in cursor.fetchall():
        key = (normalize_name(track), date, normalize_name(name))
        lookup[key] = entry_id
    print(f"Lookup: {len(lookup):,} entries")
    
    # Process months from recent to old
    total_bsp = 0
    total_prices = {tp: 0 for tp in TIME_POINTS}
    
    months_to_process = []
    for year in [2025, 2024, 2023, 2022, 2021, 2020]:
        for month in range(12, 0, -1):
            if year == 2025 and month > 12:
                continue
            months_to_process.append((year, month))
    
    for year, month in months_to_process:
        print(f"\n[{year}-{month:02d}] Getting files...")
        files = get_files_for_month(year, month)
        
        if not files:
            print(f"  No files found")
            continue
        
        print(f"  Found {len(files)} files, processing...")
        
        bsp_batch = []
        price_batches = {tp: [] for tp in TIME_POINTS}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_file, f, lookup): f for f in files}
            
            for future in as_completed(futures):
                try:
                    bsp_updates, price_updates = future.result()
                    bsp_batch.extend(bsp_updates)
                    for tp in TIME_POINTS:
                        price_batches[tp].extend(price_updates[tp])
                except Exception as e:
                    pass
        
        # Update database
        if bsp_batch:
            cursor.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", bsp_batch)
            total_bsp += len(bsp_batch)
        
        for tp in TIME_POINTS:
            if price_batches[tp]:
                cursor.executemany(f"UPDATE GreyhoundEntries SET {tp} = ? WHERE EntryID = ?", price_batches[tp])
                total_prices[tp] += len(price_batches[tp])
        
        conn.commit()
        
        print(f"  BSP: {len(bsp_batch):,} | " + " | ".join([f"{tp}: {len(price_batches[tp]):,}" for tp in list(TIME_POINTS.keys())[:3]]))
    
    conn.close()
    
    print("\n" + "="*70)
    print("IMPORT COMPLETE")
    print(f"Total BSP updates: {total_bsp:,}")
    for tp in TIME_POINTS:
        print(f"Total {tp}: {total_prices[tp]:,}")
    print("="*70)

if __name__ == "__main__":
    main()
