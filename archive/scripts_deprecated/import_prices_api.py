"""
Import BSP AND 5-min prices via Betfair Historical API
Much faster than local file processing
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

# Track mapping
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
    name = re.sub(r"['\-\.]", "", name)  # Remove apostrophes, hyphens, AND periods
    name = re.sub(r"\s+", " ", name)
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
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
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
    """Process file and return (bsp_updates, price5min_updates)"""
    bsp_updates = []
    price_updates = []
    
    content = download_file(file_path)
    if not content:
        return bsp_updates, price_updates
    
    try:
        decompressed = bz2.decompress(content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        if not lines:
            return bsp_updates, price_updates
        
        data = json.loads(lines[-1])
        if 'mc' not in data or not data['mc']:
            return bsp_updates, price_updates
        if 'marketDefinition' not in data['mc'][0]:
            return bsp_updates, price_updates
        
        md = data['mc'][0]['marketDefinition']
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        try:
            market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))
        except:
            return bsp_updates, price_updates
        
        # Get BSP and find 5-min prices
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
        runner_price5 = {sid: None for sid in runner_names}
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
                
                if diff < 120:  # Within 2 minutes of 5-min mark
                    if 'mc' in d and d['mc']:
                        for r in d['mc'][0].get('rc', []):
                            sid = r.get('id')
                            ltp = r.get('ltp')
                            if sid and ltp and sid in runner_names and diff < runner_best_diff[sid]:
                                runner_best_diff[sid] = diff
                                runner_price5[sid] = ltp
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
                
                p5 = runner_price5.get(sid)
                if p5 and p5 > 0:
                    price_updates.append((p5, entry_id))
    except:
        pass
    
    return bsp_updates, price_updates

def main():
    print("="*60)
    print("BETFAIR API - BSP + PRICE5MIN IMPORT")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure Price5Min column exists
    try:
        conn.execute("ALTER TABLE GreyhoundEntries ADD COLUMN Price5Min REAL")
        conn.commit()
        print("Added Price5Min column")
    except:
        pass
    
    # Current stats
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    initial_bsp = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Price5Min IS NOT NULL")
    initial_p5 = cursor.fetchone()[0]
    
    print(f"Current BSP: {initial_bsp:,}")
    print(f"Current Price5Min: {initial_p5:,}")
    
    # Load entries needing data
    print("\nLoading entries needing BSP or Price5Min...")
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
    
    print(f"Loaded {len(lookup):,} entries")
    
    # Process by month
    total_bsp = 0
    total_p5 = 0
    
    months_to_process = []
    for year in [2025, 2024, 2023, 2022, 2021, 2020]:
        for month in range(12, 0, -1):
            if year == 2025 and month > 11:
                continue
            months_to_process.append((year, month))
    
    for year, month in months_to_process:
        last_day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4,6,9,11] else (29 if year%4==0 else 28))
        
        from_date = datetime(year, month, 1)
        to_date = datetime(year, month, last_day)
        
        print(f"\n{year}-{month:02d}...", end=" ", flush=True)
        
        files = download_list_of_files(from_date, to_date)
        if not files:
            print("No files")
            continue
        
        print(f"{len(files)} files", end=" ", flush=True)
        
        month_bsp = []
        month_p5 = []
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_file, fp, lookup): fp for fp in files}
            for future in as_completed(futures):
                bsp_u, p5_u = future.result()
                month_bsp.extend(bsp_u)
                month_p5.extend(p5_u)
        
        print(f"-> BSP: {len(month_bsp)}, P5: {len(month_p5)}")
        
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
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"BSP: {final_bsp:,} (+{total_bsp:,})")
    print(f"Price5Min: {final_p5:,} (+{total_p5:,})")
    print("="*60)
    
    conn.close()

if __name__ == "__main__":
    main()
