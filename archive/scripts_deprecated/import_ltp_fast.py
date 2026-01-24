"""
Fast LTP Import - Extract Last Traded Price at Jump
Optimized for speed:
- Only reads last few lines of each file (LTP is near the end)
- Uses multiprocessing with larger chunks
- Batch inserts with minimal DB queries
"""
import os
import sys
import bz2
import json
import sqlite3
import re
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

def process_file_fast(file_path):
    """
    Fast extraction - only reads what we need.
    Returns: [(TrackName, RaceDate, RaceNumber, DogName, BSP, LTP), ...]
    """
    try:
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            # Read all lines but we'll only process strategically
            lines = f.readlines()
        
        if len(lines) < 2:
            return None

        # Parse LAST line for market definition and BSP
        last_data = json.loads(lines[-1])
        
        market_def = None
        if 'mc' in last_data:
            for mc in last_data['mc']:
                if 'marketDefinition' in mc:
                    market_def = mc['marketDefinition']
                    break
        
        if not market_def:
            return None

        # Quick filters
        if market_def.get('eventTypeId') != '4339':  # Not greyhound
            return None
        if market_def.get('marketType') != 'WIN':  # Not win market
            return None

        # Extract race context
        event_name = market_def.get('eventName', '')
        market_name = market_def.get('name', '')
        market_time_str = market_def.get('marketTime', '')
        
        try:
            dt = datetime.strptime(market_time_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            dt_au = dt + timedelta(hours=10)
            race_date = dt_au.strftime('%Y-%m-%d')
        except:
            return None

        track_match = re.search(r'^([^(]+)', event_name)
        track_name = track_match.group(1).strip() if track_match else event_name
        
        race_num_match = re.search(r'R(\d+)', market_name)
        if not race_num_match:
            race_num_match = re.search(r'Race\s+(\d+)', market_name)
        if not race_num_match:
            return None
        race_number = int(race_num_match.group(1))

        # Build runner ID to name mapping and get BSP
        runner_info = {}  # id -> {name, bsp}
        if 'runners' in market_def:
            for r in market_def['runners']:
                rid = r.get('id')
                name = r.get('name', '')
                bsp = r.get('bsp')
                
                # Clean name
                clean_match = re.search(r'^\d+\.\s+(.*)', name)
                clean_name = clean_match.group(1) if clean_match else name
                
                runner_info[rid] = {'name': clean_name, 'bsp': bsp, 'ltp': None}

        # Find LTP from the lines BEFORE market close
        # Scan backwards from 2nd-to-last line to find the last LTP updates
        for i in range(len(lines) - 2, max(0, len(lines) - 15), -1):
            try:
                data = json.loads(lines[i])
                if 'mc' in data:
                    for mc in data['mc']:
                        if 'rc' in mc:  # Runner Changes
                            for rc in mc['rc']:
                                rid = rc.get('id')
                                ltp = rc.get('ltp')
                                if rid in runner_info and ltp and runner_info[rid]['ltp'] is None:
                                    runner_info[rid]['ltp'] = ltp
            except:
                continue
        
        # Build results
        results = []
        for rid, info in runner_info.items():
            if info['bsp'] or info['ltp']:  # Only if we have some price data
                results.append((
                    track_name,
                    race_date,
                    race_number,
                    info['name'],
                    float(info['bsp']) if info['bsp'] else None,
                    float(info['ltp']) if info['ltp'] else None
                ))
        
        return results if results else None
        
    except Exception as e:
        return None

def ensure_ltp_column(conn):
    """Add LTP column if it doesn't exist"""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'LTP' not in columns:
        print("Adding LTP column to GreyhoundEntries...")
        cursor.execute("ALTER TABLE GreyhoundEntries ADD COLUMN LTP REAL")
        conn.commit()

def apply_updates_fast(conn, cursor, updates):
    """Batch update with both BSP and LTP"""
    success_count = 0
    
    query = """
        UPDATE GreyhoundEntries
        SET BSP = COALESCE(?, BSP), LTP = COALESCE(?, LTP)
        WHERE EntryID = (
            SELECT ge.EntryID
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE r.RaceNumber = ?
              AND rm.MeetingDate = ?
              AND g.GreyhoundName = ? COLLATE NOCASE
              AND t.TrackName LIKE ?
            LIMIT 1
        )
    """
    
    conn.execute("BEGIN TRANSACTION")
    try:
        for track_name, race_date, race_num, dog_name, bsp, ltp in updates:
            cursor.execute(query, (bsp, ltp, race_num, race_date, dog_name, f"{track_name}%"))
            if cursor.rowcount > 0:
                success_count += 1
        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    
    return success_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast LTP Import')
    parser.add_argument('--skip', type=int, default=0, help='Files to skip')
    args = parser.parse_args()
    
    print("="*60)
    print("FAST LTP IMPORT")
    print("="*60)
    
    bsp_dir = os.path.join(root_dir, 'data', 'bsp')
    print(f"Scanning {bsp_dir}...")
    
    file_list = []
    for root, dirs, files in os.walk(bsp_dir):
        for f in files:
            if f.endswith('.bz2'):
                file_list.append(os.path.join(root, f))
    
    total_found = len(file_list)
    print(f"Found {total_found} files.")
    
    if args.skip > 0:
        print(f"Skipping first {args.skip} files.")
        file_list = file_list[args.skip:]
    
    if not file_list:
        return
    
    # Setup DB
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    ensure_ltp_column(conn)
    
    # Process
    print(f"Processing with {cpu_count()} cores...")
    start_time = time.time()
    
    BATCH_SIZE = 10000  # Larger batches for speed
    buffer = []
    processed = 0
    matched = 0
    
    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(process_file_fast, file_list, chunksize=50):
            processed += 1
            if result:
                buffer.extend(result)
            
            if processed % 5000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (len(file_list) - processed) / rate / 60
                print(f"Processed {processed}/{len(file_list)} ({rate:.0f}/s) | Buffer: {len(buffer)} | ETA: {eta:.0f}min")
            
            if len(buffer) >= BATCH_SIZE:
                matched += apply_updates_fast(conn, cursor, buffer)
                buffer = []
    
    # Final batch
    if buffer:
        matched += apply_updates_fast(conn, cursor, buffer)
    
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Matched {matched} records")

if __name__ == '__main__':
    main()
