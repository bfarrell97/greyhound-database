"""
Robust BSP + Price5Min Import - Single Threaded with Retries
============================================================
Processes one file at a time to avoid API rate limiting.
Includes both BSP and 5-min-before prices.
"""
import sqlite3
import requests
import bz2
import json
import re
import time
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
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
    """Process one file and return (bsp_updates, p5_updates)"""
    bsp_updates = []
    p5_updates = []
    
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    content = download_with_retry(url, {'ssoid': SSOID})
    
    if not content:
        return [], []
    
    try:
        decompressed = bz2.decompress(content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        if not lines:
            return [], []
        
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return [], []
        if 'marketDefinition' not in data['mc'][0]:
            return [], []
        
        md = data['mc'][0]['marketDefinition']
        if md.get('marketType') != 'WIN':
            return [], []
        
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return [], []
        
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
                diff = abs(secs - 300)  # 5 min = 300 seconds
                
                if diff < 120:  # Within 2 mins of 5-min mark
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
        for sid, name in runner_names.items():
            key = (venue, market_time_str, name)
            if key in lookup:
                entry_id = lookup[key]
                
                bsp = runner_bsp.get(sid)
                if bsp:
                    bsp_updates.append((bsp, entry_id))
                
                p5 = runner_p5.get(sid)
                if p5 and p5 > 0:
                    p5_updates.append((p5, entry_id))
    except:
        pass
    
    return bsp_updates, p5_updates

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
    print("ROBUST BSP + P5 IMPORT (Single Threaded)")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Current stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    initial_bsp = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    initial_p5 = cursor.fetchone()[0]
    
    print(f"Current BSP: {initial_bsp:,}")
    print(f"Current P5: {initial_p5:,}")
    
    # Build lookup
    print("\nBuilding lookup...")
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP IS NULL OR ge.Price5Min IS NULL
    """
    
    lookup = {}
    for entry_id, track, date, dog_name in conn.execute(query).fetchall():
        bsp_track = TRACK_MAPPING.get(track, track)
        normalized = normalize_name(dog_name)
        lookup[(bsp_track, date, normalized)] = entry_id
    
    print(f"Entries needing update: {len(lookup):,}")
    
    # Process months
    total_bsp = 0
    total_p5 = 0
    
    months = []
    for year in [2025, 2024, 2023, 2022, 2021, 2020]:
        for month in range(12, 0, -1):
            if year == 2025 and month > 11:
                continue
            months.append((year, month))
    
    for year, month in months:
        files = get_files_for_month(year, month)
        if not files:
            print(f"{year}-{month:02d}: No files")
            continue
        
        print(f"\n{year}-{month:02d}: {len(files)} files", end="", flush=True)
        
        month_bsp = []
        month_p5 = []
        processed = 0
        
        for fp in files:
            bsp_u, p5_u = process_file(fp, lookup)
            month_bsp.extend(bsp_u)
            month_p5.extend(p5_u)
            processed += 1
            
            if processed % 500 == 0:
                print(f".", end="", flush=True)
        
        print(f" -> BSP: {len(month_bsp)}, P5: {len(month_p5)}")
        
        if month_bsp:
            conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ? AND BSP IS NULL", month_bsp)
        if month_p5:
            conn.executemany("UPDATE GreyhoundEntries SET Price5Min = ? WHERE EntryID = ? AND Price5Min IS NULL", month_p5)
        conn.commit()
        
        total_bsp += len(month_bsp)
        total_p5 += len(month_p5)
    
    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    final_p5 = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries")
    total = cursor.fetchone()[0]
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print(f"BSP: {final_bsp:,} / {total:,} ({final_bsp/total*100:.1f}%) [+{total_bsp:,}]")
    print(f"P5:  {final_p5:,} / {total:,} ({final_p5/total*100:.1f}%) [+{total_p5:,}]")
    print("="*70)
    
    conn.close()

if __name__ == "__main__":
    main()
