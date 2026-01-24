"""
Robust BSP + BSPPlace + Price5Min Import - 5 Threads
=====================================================
Uses Betfair Historical API to download and parse market data.
Supports WIN and PLACE markets.
"""
import sqlite3
import requests
import bz2
import json
import re
import time
import os
import sys
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.core.config import BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD

# Attempts to get session token via login
def get_session_token():
    print("Attempting automatic login to Betfair...")
    try:
        from src.integration.betfair_api import BetfairAPI
        api = BetfairAPI(BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD)
        return api.login()
    except Exception as e:
        print(f"Automatic login failed: {e}")
        return None

# Global SSOID placeholder
SSOID = None
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

def normalize_name(name):
    if not name: return ""
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
            elif resp.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)
        except:
            time.sleep(1)
    return None

def process_file(file_path, lookup):
    """Process one file and return (bsp_updates, p5_updates, place_updates)"""
    bsp_updates = []
    p5_updates = []
    place_updates = []
    
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    content = download_with_retry(url, {'ssoid': SSOID})
    
    if not content:
        return [], [], []
    
    try:
        decompressed = bz2.decompress(content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        if not lines:
            return [], [], []
        
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return [], [], []
        if 'marketDefinition' not in data['mc'][0]:
            return [], [], []
        
        md = data['mc'][0]['marketDefinition']
        mtype = md.get('marketType')
        if mtype not in ['WIN', 'PLACE']:
            return [], [], []
        
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return [], [], []
        
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
        
        # Find 5-min prices
        runner_p5 = {sid: None for sid in runner_names}
        runner_best_diff = {sid: float('inf') for sid in runner_names}
        
        for line in lines[:-1]:
            try:
                d = json.loads(line)
                pt = d.get('pt')
                if not pt:
                    continue
                pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                secs = (market_time - pub_time).total_seconds()
                diff = abs(secs - 300)
                
                if diff < 120:
                    if 'mc' in d and d['mc']:
                        for r in d['mc'][0].get('rc', []):
                            sid = r.get('id')
                            ltp = r.get('ltp')
                            if sid and ltp and sid in runner_names and diff < runner_best_diff[sid]:
                                runner_best_diff[sid] = diff
                                runner_p5[sid] = ltp
            except:
                continue
        
        # Create updates
        debug_path = "import_debug.log"
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            
            # Log all to file for analysis
            with open(debug_path, "a") as f:
                f.write(f"Venue: {venue} | Date: {market_time_str} | Name: {name} | Match: {key in lookup}\n")

            if key in lookup:
                entry_id = lookup[key]
                bsp = runner_bsp.get(sid)
                if bsp:
                    if mtype == 'WIN':
                        bsp_updates.append((bsp, entry_id))
                    else:
                        place_updates.append((bsp, entry_id))
                
                if mtype == 'WIN':
                    p5 = runner_p5.get(sid)
                    if p5 and p5 > 0:
                        p5_updates.append((p5, entry_id))

    except Exception:
        pass
    
    return bsp_updates, p5_updates, place_updates

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
        "marketTypesCollection": ["WIN", "PLACE"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error getting files: {e}")
    return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Import BSP from Betfair History (API)')
    parser.add_argument('--ssoid', type=str, help='Manual Betfair SSOID/Session Token')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - only scan files')
    args = parser.parse_args()

    global SSOID
    if args.ssoid:
        SSOID = args.ssoid
        print(f"Using manual SSOID: {SSOID[:10]}...")
    else:
        SSOID = get_session_token()
        if not SSOID:
            SSOID = 'XTbdQ3x85MZl2+Xj92Oj2DuTic9C6OC+f8GR/6vb7UQ='
            print(f"Using fallback SSOID: {SSOID[:10]}...")

    if not SSOID:
        print("[ERROR] No session token available. Provide --ssoid or check credentials.")
        return

    print("="*70)
    print("ROBUST WIN + PLACE BSP + P5 IMPORT (5 Threads, Feb 2025 backwards)")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Current stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    initial_bsp = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSPPlace IS NOT NULL")
    initial_place = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    initial_p5 = cursor.fetchone()[0]
    
    print(f"Current Win BSP:   {initial_bsp:,}")
    print(f"Current Place BSP: {initial_place:,}")
    print(f"Current P5:        {initial_p5:,}")
    
    # Build lookup
    print("\nBuilding lookup...")
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE (ge.BSP IS NULL OR ge.Price5Min IS NULL OR ge.BSPPlace IS NULL)
      AND rm.MeetingDate < '2026-01-01'
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        bsp_track = TRACK_MAPPING.get(track, track)
        normalized = normalize_name(dog_name)
        lookup[(bsp_track, date, normalized)] = entry_id
    
    print(f"Entries needing update: {len(lookup):,}")
    
    # Process months from Feb 2025 backwards
    total_bsp = 0
    total_place = 0
    total_p5 = 0
    
    months = []
    for year in range(2025, 2018, -1):
        for month in range(12, 0, -1):
            if year == 2025 and month > 11:
                continue
            months.append((year, month))
    
    for year, month in months:
        files = get_files_for_month(year, month)
        if not files:
            continue
        
        print(f"\n{year}-{month:02d}: {len(files)} files", end="", flush=True)
        
        month_bsp = []
        month_p5 = []
        month_place = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_file, fp, lookup): fp for fp in files}
            
            for future in as_completed(futures):
                try:
                    bsp_u, p5_u, place_u = future.result()
                    month_bsp.extend(bsp_u)
                    month_p5.extend(p5_u)
                    month_place.extend(place_u)
                except Exception:
                    pass
                processed += 1
                print(".", end="", flush=True)
        
        print(f" -> Win: {len(month_bsp)}, Place: {len(month_place)}, P5: {len(month_p5)}")
    
    conn.close()

if __name__ == "__main__":
    main()
