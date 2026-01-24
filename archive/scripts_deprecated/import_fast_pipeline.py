"""
Fast Import Pipeline using betfair_data (Rust)
==============================================
1. Downloads monthly data (multi-threaded) to temp dir.
2. Parses using betfair_data (Rust binding) at ~500k/sec.
3. Updates DB using bulk insert.
4. Cleans up temp files.

Author: Antigravity
"""
import os
import shutil
import sqlite3
import requests
import json
import time
import re
import betfair_data as bfd
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_URL = "https://historicdata.betfair.com/api/"
SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0=" # Updated from user
TEMP_DIR = "temp_import_data"
DB_PATH = "greyhound_racing.db"
MAX_THREADS = 10 

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

# --- 1. Downloader Logic ---

def get_files_for_month(year, month):
    last_day = 31 if month in [1,3,5,7,8,10,12] else (30 if month in [4,6,9,11] else (29 if year%4==0 else 28))
    url = BASE_URL + 'DownloadListOfFiles'
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": 1, "fromMonth": month, "fromYear": year,
        "toDay": last_day, "toMonth": month, "toYear": year,
        "marketTypesCollection": ["WIN"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching file list: {e}")
    return []

def download_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    local_path = os.path.join(output_dir, filename)
    
    # Skip if exists
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    headers = {'ssoid': SSOID}
    
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
        except:
            time.sleep(1)
    return None

def download_month_batch(year, month):
    month_dir = os.path.join(TEMP_DIR, f"{year}_{month:02d}")
    os.makedirs(month_dir, exist_ok=True)
    
    print(f"[{year}-{month:02d}] Fetching file list...")
    files = get_files_for_month(year, month)
    if not files:
        print("  No files found.")
        return None
        
    print(f"  Downloading {len(files)} files to {month_dir}...")
    
    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(download_file, f, month_dir): f for f in files}
        completed = 0
        for future in as_completed(futures):
            path = future.result()
            if path:
                downloaded_paths.append(path)
            completed += 1
            if completed % 500 == 0:
                print(f"    {completed}/{len(files)} downloaded...")
                
    return month_dir, downloaded_paths

# --- 2. Parser Logic (betfair_data) ---

def normalize_name(name):
    if not name: return ''
    match = re.match(r'\d+\.\s*(.+)', name)
    if match: name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    for old, new in TRACK_MAPPING.items():
        name = name.replace(old, new)
    return name

def parse_and_update(data_dir):
    print(f"  Parsing markets in {data_dir} using betfair_data...")
    
    # Load Lookup Table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ge.EntryID, t.TrackName, rm.MeetingDate, g.GreyhoundName
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    """)
    # Logic: Key = (Venue, Name), Value = List of (DateStr, EntryID)
    lookup = {} 
    for eid, trk, dat, nam in cursor.fetchall():
        key = (normalize_name(trk), normalize_name(nam))
        if key not in lookup:
            lookup[key] = []
        lookup[key].append((dat, eid))
    conn.close()
    
    print(f"  Lookup loaded: {len(lookup):,} unique venue/dog pairs")
    
    # Process Files
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bz2')]
    
    batch_bsp = []
    time_cols = ['Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']
    time_windows = [3600, 1800, 900, 600, 300, 120, 60]
    
    batch_prices = {col: [] for col in time_cols}
    
    market_count = 0
    import logging
    logging.basicConfig(level=logging.ERROR)
    from datetime import datetime
    
    try:
        for file_obj in bfd.Files(paths):
            # Per-market state
            market_state = {
                'venue': None,
                'date': None,       # Str YYYY-MM-DD (UTC)
                'market_time': None,# datetime (UTC)
                'runner_map': None, # SelectionID -> EntryID
                'snapshots_taken': {col: False for col in time_cols},
                'valid': True
            }
            
            # Iterate UPDATES in the file
            for update in file_obj:
                if not market_state['valid']:
                    break 

                # 1. Init / Metadata 
                if not market_state['venue']:
                    if hasattr(update, 'venue') and update.venue:
                         market_state['venue'] = normalize_name(update.venue)
                    
                    if hasattr(update, 'market_time') and update.market_time:
                         market_state['market_time'] = update.market_time
                         market_state['date'] = update.market_time.strftime('%Y-%m-%d')

                    if not market_state['venue']:
                        continue
                        
                # 2. Runners Mapping (FLEXIBLE DATE MATCHING)
                if not market_state['runner_map'] and hasattr(update, 'runners') and update.runners:
                    m_venue = market_state['venue']
                    # m_time is UTC datetime
                    if not market_state['market_time']: continue
                    
                    m_date_obj = market_state['market_time'].date()
                    
                    rmap = {}
                    for runner in update.runners:
                        name = normalize_name(runner.name)
                        key = (m_venue, name)
                        
                        if key in lookup:
                            # Iterate candidates to find date match (+/- 1 day)
                            candidates = lookup[key]
                            best_match_eid = None
                            
                            for db_date_str, eid in candidates:
                                try:
                                    db_date_obj = datetime.strptime(db_date_str, '%Y-%m-%d').date()
                                    diff = abs((db_date_obj - m_date_obj).days)
                                    if diff <= 1:
                                        best_match_eid = eid
                                        break # Found a match
                                except:
                                    continue
                            
                            if best_match_eid:
                                rmap[runner.selection_id] = best_match_eid
                            
                    if rmap:
                        market_state['runner_map'] = rmap
                        market_count += 1
                        # print(f"Mapped {len(rmap)} runners for {m_venue}...") # Debug
                    else:
                        pass # No match

                # 3. Process Update
                if not market_state['runner_map']:
                     continue
                     
                rmap = market_state['runner_map']
                m_time = market_state['market_time']
                
                if hasattr(update, 'publish_time'):
                    pt = update.publish_time
                else:
                    continue
                    
                seconds_out = (m_time - pt).total_seconds()
                
                # Check BSP & Prices
                for runner in update.runners:
                    sid = runner.selection_id
                    if sid not in rmap: continue
                    entry_id = rmap[sid]
                    
                    # Capture BSP
                    if hasattr(runner, 'bsp') and runner.bsp:
                         batch_bsp.append((runner.bsp, entry_id))

                    # Capture LTP for Snapshots
                    ltp = getattr(runner, 'ltp', None)
                    if not ltp: continue
                    
                    for w, col in zip(time_windows, time_cols):
                        if not market_state['snapshots_taken'][col] and seconds_out <= w:
                            market_state['snapshots_taken'][col] = True
                            
                            # Grab LTP for ALL runners?
                            # Optim: We are inside runner loop, but snapshot flag is per market?
                            # ERROR: logic flaw.
                            # If we set flag=True inside runner loop, we only grab THIS runner.
                            # We must iterate ALL runners if the snapshot triggers?
                            # Or just append for THIS runner?
                            # Answer: Append for THIS runner.
                            # BUT we must set flag only after checking ALL runners?
                            # No, if we set flag, next runner loop sees it set.
                            pass
                
                # Correct Snapshot Logic:
                # We need to grab ALL prices at the moment the snapshot trigger happens.
                # BUT update.runners listing contains all of them?
                # Yes.
                # So verify:
                
                # Iterate snapshots triggers
                for w, col in zip(time_windows, time_cols):
                     if not market_state['snapshots_taken'][col] and seconds_out <= w:
                        market_state['snapshots_taken'][col] = True
                        
                        # Grab prices for ALL mapped runners in this update
                        for runner in update.runners:
                            sid = runner.selection_id
                            if sid in rmap:
                                ltp = getattr(runner, 'ltp', None)
                                if ltp:
                                    batch_prices[col].append((ltp, rmap[sid]))
            
            if market_count % 100 == 0:
                print(f"    Parsed {market_count} markets...", end='\r')

    except Exception as e:
        print(f"Error in betfair_data loop: {e}")

    print(f"\n  Parsing done. Markets Matched: {market_count}")
    
    # Batch Update DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Dedupe BSPs (Use simple dict)
    bsp_dict = {eid: b for b, eid in batch_bsp}
    final_bsp = [(b, eid) for eid, b in bsp_dict.items()]
    
    if final_bsp:
        c.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", final_bsp)
        print(f"  Updated {len(final_bsp)} BSPs")
        
    for col in time_cols:
        updates = batch_prices[col]
        # Dedupe (Keep First encountered)
        # Updates list is chronological? Market updates are chronological.
        # So we captured the FIRST time seconds_out <= 600.
        # This is correct for T-10m.
        # But we might have duplicates due to multiple markets matching same race?
        # Unlikely.
        # Use dict to be safe (Keep Last or First? First is T-60m exactly. Last might be T-59m).
        # We want the one closest to boundary?
        # The logic `if not snapshot_taken` ensures we interpret only one update per market.
        
        # Safe dedupe by ID
        # Note: If multiple markets map to same ID (e.g. duplicate market files), we might have conflict.
        # Keep First.
        p_dict = {} # eid -> price
        for p, eid in updates:
            if eid not in p_dict:
                p_dict[eid] = p
        
        final_updates = [(p, eid) for eid, p in p_dict.items()]

        if final_updates:
            c.executemany(f"UPDATE GreyhoundEntries SET {col} = ? WHERE EntryID = ?", final_updates)
            print(f"  Updated {len(final_updates)} {col}")
            
    conn.commit()
    conn.close()

# --- 3. Manager ---

def main():
    print("="*70)
    print("FAST IMPORT PIPELINE")
    print("Using betfair_data (Rust) + Threaded Download")
    print("="*70)
    
    # Ensure columns
    conn = sqlite3.connect(DB_PATH)
    for col in ['Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']:
        try:
            conn.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {col} REAL")
        except: pass
    conn.close()
    
    # Process
    for year in [2025]: # Test with 2025 first
        for month in range(12, 0, -1):
            # Only if files exist? 
            # We assume Aug 2025 works.
            # Try Oct (already done? Do it again to be fast?)
            # Let's try July 2025 (New data).
            if year == 2025 and month > 12: continue
            
            res = download_month_batch(year, month)
            if res:
                month_dir, paths = res
                if paths:
                    parse_and_update(month_dir)
                
                # Cleanup
                print(f"  Cleaning up {month_dir}...")
                shutil.rmtree(month_dir)
                
if __name__ == "__main__":
    main()
