"""
FAST LTP vs SP Comparison
Pre-loads SP data into memory for instant lookups
"""
import os
import bz2
import json
import sqlite3
import re
import random
from datetime import datetime, timedelta

DB_PATH = 'greyhound_racing.db'

def extract_ltp_from_file(file_path):
    """Extract LTP and BSP from a single file - returns list of dicts"""
    try:
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None

        last_data = json.loads(lines[-1])
        
        market_def = None
        if 'mc' in last_data:
            for mc in last_data['mc']:
                if 'marketDefinition' in mc:
                    market_def = mc['marketDefinition']
                    break
        
        if not market_def:
            return None

        if market_def.get('eventTypeId') != '4339':
            return None
        if market_def.get('marketType') != 'WIN':
            return None

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
            return None
        race_number = int(race_num_match.group(1))

        runner_info = {}
        if 'runners' in market_def:
            for r in market_def['runners']:
                rid = r.get('id')
                name = r.get('name', '')
                bsp = r.get('bsp')
                clean_match = re.search(r'^\d+\.\s+(.*)', name)
                clean_name = clean_match.group(1).upper() if clean_match else name.upper()
                runner_info[rid] = {'name': clean_name, 'bsp': bsp, 'ltp': None}

        for i in range(len(lines) - 2, max(0, len(lines) - 15), -1):
            try:
                data = json.loads(lines[i])
                if 'mc' in data:
                    for mc in data['mc']:
                        if 'rc' in mc:
                            for rc in mc['rc']:
                                rid = rc.get('id')
                                ltp = rc.get('ltp')
                                if rid in runner_info and ltp and runner_info[rid]['ltp'] is None:
                                    runner_info[rid]['ltp'] = ltp
            except:
                continue
        
        results = []
        for rid, info in runner_info.items():
            if info['ltp']:
                results.append({
                    'track': track_name.upper(),
                    'date': race_date,
                    'race': race_number,
                    'dog': info['name'],
                    'bsp': info['bsp'],
                    'ltp': info['ltp']
                })
        return results
        
    except:
        return None

def main():
    import numpy as np
    
    print("="*60)
    print("FAST LTP vs SP COMPARISON")
    print("="*60)
    
    # Step 1: Load ALL SP data into memory for fast lookup
    print("\nLoading SP data from database...")
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            UPPER(t.TrackName) as track,
            rm.MeetingDate as date,
            r.RaceNumber as race,
            UPPER(g.GreyhoundName) as dog,
            ge.StartingPrice as sp,
            ge.Position
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= '2024-01-01'
          AND ge.StartingPrice IS NOT NULL
    """
    import pandas as pd
    sp_df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(sp_df)} SP records into memory")
    
    # Create lookup dict (using dict comprehension - last value wins for duplicates)
    sp_lookup = {}
    for _, row in sp_df.iterrows():
        key = f"{row['date']}_{row['race']}_{row['dog']}"
        sp_lookup[key] = {'sp': row['sp'], 'Position': row['Position']}
    
    # Step 2: Sample and extract LTP from files
    print("\nSampling Betfair files...")
    bsp_dir = 'data/bsp'
    all_files = []
    for root, dirs, files in os.walk(bsp_dir):
        for f in files:
            if f.endswith('.bz2'):
                all_files.append(os.path.join(root, f))
    
    sample_files = random.sample(all_files, min(500, len(all_files)))
    print(f"Processing {len(sample_files)} files...")
    
    ltp_data = []
    for i, f in enumerate(sample_files):
        result = extract_ltp_from_file(f)
        if result:
            ltp_data.extend(result)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(sample_files)} files, {len(ltp_data)} LTP records")
    
    print(f"\nExtracted {len(ltp_data)} LTP records")
    
    # Step 3: Match with SP using memory lookup
    print("Matching with SP data...")
    matches = []
    for item in ltp_data:
        # Try exact match first
        key = f"{item['date']}_{item['race']}_{item['dog']}"
        if key in sp_lookup:
            sp_data = sp_lookup[key]
            sp_str = sp_data['sp']
            try:
                sp = float(sp_str.replace('$', '').replace('F', '').strip())
                matches.append({
                    'ltp': item['ltp'],
                    'bsp': item['bsp'],
                    'sp': sp,
                    'is_winner': sp_data['Position'] == '1'
                })
            except:
                pass
    
    print(f"Matched {len(matches)} records")
    
    if len(matches) < 10:
        print("Not enough matches for analysis!")
        return
    
    # Step 4: Analysis
    ltps = np.array([m['ltp'] for m in matches])
    sps = np.array([m['sp'] for m in matches])
    bsps = np.array([m['bsp'] for m in matches if m['bsp']])
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nSample Size: {len(matches)}")
    print(f"\nAverage Prices:")
    print(f"  SP:  ${np.mean(sps):.2f}")
    print(f"  LTP: ${np.mean(ltps):.2f}")
    if len(bsps) > 0:
        print(f"  BSP: ${np.mean(bsps):.2f}")
    
    ltp_lower = (ltps < sps).sum()
    sp_lower = (sps < ltps).sum()
    
    print(f"\nFor LAYING (lower = better):")
    print(f"  LTP < SP: {ltp_lower} times ({ltp_lower/len(matches)*100:.1f}%) - LTP wins")
    print(f"  SP < LTP: {sp_lower} times ({sp_lower/len(matches)*100:.1f}%) - SP wins")
    
    # Lay candidates analysis
    lay_pool = [m for m in matches if 1.50 <= m['sp'] <= 3.00]
    if lay_pool:
        print(f"\n--- LAY CANDIDATES ($1.50-$3.00 SP) ---")
        print(f"Count: {len(lay_pool)}")
        
        lay_ltps = np.array([m['ltp'] for m in lay_pool])
        lay_sps = np.array([m['sp'] for m in lay_pool])
        
        print(f"Avg SP:  ${np.mean(lay_sps):.2f}")
        print(f"Avg LTP: ${np.mean(lay_ltps):.2f}")
        
        sp_liability = (lay_sps - 1) * 100
        ltp_liability = (lay_ltps - 1) * 100
        
        print(f"\nAvg Liability at $100 stake:")
        print(f"  At SP:  ${np.mean(sp_liability):.0f}")
        print(f"  At LTP: ${np.mean(ltp_liability):.0f}")
        
        diff = np.mean(sp_liability) - np.mean(ltp_liability)
        if diff > 0:
            print(f"  LTP SAVES: ${diff:.0f} per bet (BETTER for laying)")
        else:
            print(f"  LTP COSTS: ${-diff:.0f} more per bet (WORSE for laying)")
        
        ltp_lower_lay = (lay_ltps < lay_sps).sum()
        print(f"\nLTP lower than SP in {ltp_lower_lay}/{len(lay_pool)} cases ({ltp_lower_lay/len(lay_pool)*100:.1f}%)")

if __name__ == "__main__":
    main()
