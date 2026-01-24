"""
Quick LTP vs SP Comparison
Samples files to compare Last Traded Price with Starting Price
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
    """Extract LTP and BSP from a single file"""
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

        # Get BSP from last line
        runner_info = {}
        if 'runners' in market_def:
            for r in market_def['runners']:
                rid = r.get('id')
                name = r.get('name', '')
                bsp = r.get('bsp')
                clean_match = re.search(r'^\d+\.\s+(.*)', name)
                clean_name = clean_match.group(1) if clean_match else name
                runner_info[rid] = {'name': clean_name, 'bsp': bsp, 'ltp': None}

        # Find last LTP before market close
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
                    'track': track_name,
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
    print("="*60)
    print("LTP vs SP QUICK COMPARISON")
    print("="*60)
    
    # Sample files
    bsp_dir = 'data/bsp'
    all_files = []
    for root, dirs, files in os.walk(bsp_dir):
        for f in files:
            if f.endswith('.bz2'):
                all_files.append(os.path.join(root, f))
    
    print(f"Found {len(all_files)} files, sampling 1000...")
    sample_files = random.sample(all_files, min(1000, len(all_files)))
    
    # Extract LTP data
    ltp_data = []
    for i, f in enumerate(sample_files):
        result = extract_ltp_from_file(f)
        if result:
            ltp_data.extend(result)
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/1000 files, found {len(ltp_data)} LTP records")
    
    print(f"\nExtracted {len(ltp_data)} LTP records")
    
    # Match with SP from database
    conn = sqlite3.connect(DB_PATH)
    
    matches = []
    for item in ltp_data:
        query = """
            SELECT ge.StartingPrice, ge.Position
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
        """
        result = conn.execute(query, (item['race'], item['date'], item['dog'], f"{item['track']}%")).fetchone()
        if result and result[0]:
            sp_str = result[0]
            try:
                sp = float(sp_str.replace('$', '').replace('F', '').strip())
                matches.append({
                    'dog': item['dog'],
                    'ltp': item['ltp'],
                    'bsp': item['bsp'],
                    'sp': sp,
                    'is_winner': result[1] == '1'
                })
            except:
                pass
    
    conn.close()
    
    print(f"\nMatched {len(matches)} records with SP data")
    
    if not matches:
        print("No matches found!")
        return
    
    # Analysis
    import numpy as np
    
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
    
    # For laying, lower is better
    ltp_better = (ltps < sps).sum()
    sp_better = (sps < ltps).sum()
    
    print(f"\nFor LAYING (lower is better):")
    print(f"  LTP < SP: {ltp_better} times ({ltp_better/len(matches)*100:.1f}%)")
    print(f"  SP < LTP: {sp_better} times ({sp_better/len(matches)*100:.1f}%)")
    
    # Liability comparison (for lay bets at $1.50-$3.00)
    lay_pool = [m for m in matches if 1.50 <= m['sp'] <= 3.00]
    if lay_pool:
        print(f"\n--- LAY CANDIDATES ($1.50-$3.00 SP) ---")
        print(f"Count: {len(lay_pool)}")
        
        lay_ltps = np.array([m['ltp'] for m in lay_pool])
        lay_sps = np.array([m['sp'] for m in lay_pool])
        
        print(f"Avg SP:  ${np.mean(lay_sps):.2f}")
        print(f"Avg LTP: ${np.mean(lay_ltps):.2f}")
        
        # Liability at $100 stake
        sp_liability = (lay_sps - 1) * 100
        ltp_liability = (lay_ltps - 1) * 100
        
        print(f"\nAvg Liability at $100 stake:")
        print(f"  At SP:  ${np.mean(sp_liability):.0f}")
        print(f"  At LTP: ${np.mean(ltp_liability):.0f}")
        print(f"  Savings: ${np.mean(sp_liability) - np.mean(ltp_liability):.0f} per bet")
        
        ltp_lower = (lay_ltps < lay_sps).sum()
        print(f"\nLTP lower than SP in {ltp_lower}/{len(lay_pool)} cases ({ltp_lower/len(lay_pool)*100:.1f}%)")

if __name__ == "__main__":
    main()
