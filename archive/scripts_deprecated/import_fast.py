"""
Fast BSP + P5 Import with 5 Threads
====================================
Starts from Feb 2025 and works backwards.
Uses thread pool to speed up processing.
"""
import sqlite3
import requests
import bz2
import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import sys

# Betfair Historical Data API
SSOID = "z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc="
BASE_URL = "https://historicdata.betfair.com/api/"

# Track mapping
TRACK_MAPPING = {
    'BET DELUXE CAPALABA': 'CAPALABA',
    'TAB PHOENIX PARK': 'PHOENIX PARK',
    'TAB SANDOWN PARK': 'SANDOWN PARK',
    'TAB WENTWORTH PARK': 'WENTWORTH PARK',
    'TAB SHEPPARTON': 'SHEPPARTON',
    'TAB GREYHOUNDS BENDIGO': 'BENDIGO',
    'TAB HEALESVILLE': 'HEALESVILLE',
    'TAB THE MEADOWS': 'THE MEADOWS',
    'TAB TRARALGON': 'TRARALGON',
    'TAB SALE': 'SALE',
}

def normalize_name(name):
    """Normalize dog name for matching"""
    if not name:
        return ""
    name = name.upper().strip()
    name = name.replace("'", "").replace("-", "").replace(".", "")
    return name

# Thread lock for DB writes
db_lock = threading.Lock()

def download_file(file_path, session):
    """Download and parse a single file"""
    try:
        from urllib.parse import quote
        encoded_path = quote(file_path, safe='')
        url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
        headers = {'ssoid': SSOID}
        resp = session.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            return None
        data = bz2.decompress(resp.content).decode('utf-8')
        
        results = []
        venue = None
        market_time_str = None
        runner_bsp = {}
        runner_names = {}
        runner_5min = {}
        
        for line in data.strip().split('\n'):
            try:
                obj = json.loads(line)
            except:
                continue
                
            if 'mc' not in obj:
                continue
                
            for mc in obj['mc']:
                if 'marketDefinition' in mc:
                    md = mc['marketDefinition']
                    venue = md.get('venue', '').upper()
                    if venue in TRACK_MAPPING:
                        venue = TRACK_MAPPING[venue]
                    
                    mt = md.get('marketTime', '')
                    if mt:
                        try:
                            dt = datetime.fromisoformat(mt.replace('Z', '+00:00'))
                            market_time_str = dt.strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    for runner in md.get('runners', []):
                        sid = runner.get('id')
                        name = runner.get('name', '')
                        if sid and name:
                            runner_names[sid] = normalize_name(name)
                
                # BSP extraction
                if 'rc' in mc:
                    for rc in mc['rc']:
                        sid = rc.get('id')
                        sp = rc.get('spb')
                        if sid and sp:
                            runner_bsp[sid] = sp
                
                # 5-min price extraction
                pt = obj.get('pt')
                if pt and market_time_str and 'rc' in mc:
                    try:
                        current_dt = datetime.fromtimestamp(pt/1000)
                        race_date = market_time_str.split(' ')[0] if ' ' in market_time_str else market_time_str
                        race_dt = datetime.fromisoformat(race_date + "T00:00:00")
                        
                        # Around 5 mins before = -360 to -240 seconds
                        if 'marketTime' in mc.get('marketDefinition', {}):
                            mt = mc['marketDefinition']['marketTime']
                            race_dt = datetime.fromisoformat(mt.replace('Z', '+00:00')).replace(tzinfo=None)
                            diff = (race_dt - current_dt).total_seconds()
                            
                            if 240 <= diff <= 360:  # 4-6 mins before
                                for rc in mc['rc']:
                                    sid = rc.get('id')
                                    ltp = rc.get('ltp')
                                    if sid and ltp and sid not in runner_5min:
                                        runner_5min[sid] = ltp
                    except:
                        pass
        
        if venue and market_time_str:
            for sid, name in runner_names.items():
                bsp = runner_bsp.get(sid)
                p5 = runner_5min.get(sid)
                if bsp or p5:
                    results.append({
                        'venue': venue,
                        'date': market_time_str,
                        'name': name,
                        'bsp': bsp,
                        'p5': p5
                    })
        
        return results
    except Exception as e:
        return None

def process_month(year, month, lookup, session):
    """Process a single month"""
    # Calculate last day of month
    last_day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4,6,9,11] else (29 if year%4==0 else 28))
    
    # Get file list using POST with JSON (like import_robust.py)
    url = f"{BASE_URL}DownloadListOfFiles"
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
        resp = session.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"\n{year}-{month:02d}: HTTP {resp.status_code}")
            return 0, 0
        files = resp.json()
    except Exception as e:
        print(f"\n{year}-{month:02d}: Request error: {e}")
        return 0, 0
    
    if not files or not isinstance(files, list):
        print(f"\n{year}-{month:02d}: No files (response type: {type(files)})")
        return 0, 0
    
    print(f"\n{year}-{month:02d}: {len(files)} files", end='', flush=True)
    
    bsp_count = 0
    p5_count = 0
    updates = []
    
    # Use 5 threads for faster download
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_file, f, session): f for f in files}
        
        for i, future in enumerate(as_completed(futures)):
            if (i+1) % 500 == 0:
                print('.', end='', flush=True)
            
            results = future.result()
            if not results:
                continue
            
            for r in results:
                key = (r['venue'], r['date'], r['name'])
                if key in lookup:
                    entry_id = lookup[key]
                    updates.append((r['bsp'], r['p5'], entry_id))
                    if r['bsp']:
                        bsp_count += 1
                    if r['p5']:
                        p5_count += 1
    
    # Batch update DB
    if updates:
        with db_lock:
            conn = sqlite3.connect('greyhound_racing.db')
            cursor = conn.cursor()
            for bsp, p5, entry_id in updates:
                if bsp and p5:
                    cursor.execute("UPDATE GreyhoundEntries SET BSP=?, Price5Min=? WHERE EntryID=?", (bsp, p5, entry_id))
                elif bsp:
                    cursor.execute("UPDATE GreyhoundEntries SET BSP=? WHERE EntryID=?", (bsp, entry_id))
                elif p5:
                    cursor.execute("UPDATE GreyhoundEntries SET Price5Min=? WHERE EntryID=?", (p5, entry_id))
            conn.commit()
            conn.close()
    
    print(f" -> BSP: {bsp_count}, P5: {p5_count}")
    return bsp_count, p5_count

def main():
    print("="*70)
    print("FAST BSP + P5 IMPORT (5 Threads, Feb 2025 backwards)")
    print("="*70)
    
    # Check current coverage
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    current_bsp = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    current_p5 = cursor.fetchone()[0]
    print(f"Current BSP: {current_bsp:,}")
    print(f"Current P5: {current_p5:,}")
    
    # Build lookup
    print("\nBuilding lookup...")
    cursor.execute("""
        SELECT ge.EntryID, t.TrackName, rm.MeetingDate, g.GreyhoundName
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE ge.BSP IS NULL
          AND rm.MeetingDate < '2025-03-01'
    """)
    
    lookup = {}
    for row in cursor.fetchall():
        entry_id, track, date, dog = row
        key = (track.upper(), date, normalize_name(dog))
        lookup[key] = entry_id
    
    conn.close()
    print(f"Entries to update: {len(lookup):,}")
    
    # Process months backwards from Feb 2025
    session = requests.Session()
    total_bsp = 0
    total_p5 = 0
    
    # Feb 2025 to Jan 2020 (backwards)
    months = []
    for year in range(2025, 2019, -1):
        end_month = 2 if year == 2025 else 12  # Start from Feb 2025
        for month in range(end_month, 0, -1):
            months.append((year, month))
    
    print(f"\nProcessing {len(months)} months: {months[0]} to {months[-1]}")
    
    for year, month in months:
        try:
            bsp, p5 = process_month(year, month, lookup, session)
            total_bsp += bsp
            total_p5 += p5
        except Exception as e:
            print(f"\nError processing {year}-{month:02d}: {e}")
    
    print("\n" + "="*70)
    print(f"COMPLETE! Added BSP: {total_bsp:,}, P5: {total_p5:,}")
    print("="*70)

if __name__ == "__main__":
    main()
