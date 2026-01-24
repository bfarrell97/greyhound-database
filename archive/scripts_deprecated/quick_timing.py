"""
Quick timing analysis using just September 2025 data
Prints progress to confirm not frozen, outputs results fast
"""
import bz2
import json
import os
import statistics
from datetime import datetime
from collections import defaultdict

def normalize_name(name):
    import re
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    return name.upper().replace("'", "").replace("-", "").strip()

def main():
    print("="*60)
    print("QUICK TIMING ANALYSIS - September 2025 only")
    print("="*60)
    
    lay_data = defaultdict(list)   # BSP < 2.25
    back_data = defaultdict(list)  # BSP 3-8
    time_buckets = [600, 300, 120, 60, 30, 10]
    
    base_folder = r'data\bsp\1\BASIC\2025\Sep'
    files_done = 0
    win_markets = 0
    
    for day in os.listdir(base_folder):
        day_path = os.path.join(base_folder, day)
        if not os.path.isdir(day_path):
            continue
        
        for event in os.listdir(day_path):
            event_path = os.path.join(day_path, event)
            if not os.path.isdir(event_path):
                continue
            
            for f in os.listdir(event_path):
                if not f.endswith('.bz2'):
                    continue
                
                files_done += 1
                
                try:
                    with bz2.open(os.path.join(event_path, f), 'rt') as fh:
                        lines = fh.readlines()
                    
                    if len(lines) < 2:
                        continue
                    
                    data = json.loads(lines[-1])
                    if 'mc' not in data or not data['mc']:
                        continue
                    
                    mc = data['mc'][0]
                    if 'marketDefinition' not in mc:
                        continue
                    
                    md = mc['marketDefinition']
                    if md.get('marketType') != 'WIN':
                        continue
                    
                    win_markets += 1
                    
                    market_time_str = md.get('marketTime', '')
                    try:
                        market_time = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
                    except:
                        continue
                    
                    # Get BSP
                    runner_bsp = {}
                    for r in md.get('runners', []):
                        bsp = r.get('bsp')
                        if bsp and bsp > 0:
                            runner_bsp[r.get('id')] = bsp
                    
                    # Get prices over time
                    runner_prices = {sid: {} for sid in runner_bsp}
                    
                    for line in lines[:-1]:
                        try:
                            d = json.loads(line)
                            pt = d.get('pt')
                            if not pt:
                                continue
                            pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                            secs = (market_time - pub_time).total_seconds()
                            
                            if 'mc' in d and d['mc']:
                                for r in d['mc'][0].get('rc', []):
                                    sid = r.get('id')
                                    ltp = r.get('ltp')
                                    if sid and ltp and sid in runner_bsp:
                                        runner_prices[sid][secs] = ltp
                        except:
                            continue
                    
                    # Bucket prices
                    for sid, bsp in runner_bsp.items():
                        prices = runner_prices.get(sid, {})
                        for bucket in time_buckets:
                            # Find closest to bucket
                            best = None
                            best_diff = 30
                            for t, p in prices.items():
                                diff = abs(t - bucket)
                                if diff < best_diff:
                                    best_diff = diff
                                    best = p
                            
                            if best and best > 0:
                                ratio = best / bsp
                                if bsp < 2.25:
                                    lay_data[bucket].append(ratio)
                                elif 3 <= bsp <= 8:
                                    back_data[bucket].append(ratio)
                
                except Exception as e:
                    continue
                
                if files_done % 5000 == 0:
                    print(f"  {files_done} files, {win_markets} WIN markets...", flush=True)
    
    print(f"\nDone: {files_done} files, {win_markets} WIN markets")
    
    print("\n" + "="*60)
    print("LAY (BSP < $2.25) - Best time to LAY")
    print("="*60)
    for b in sorted(time_buckets, reverse=True):
        if lay_data[b]:
            avg = statistics.mean(lay_data[b])
            print(f"  {b//60}m {b%60}s before: {len(lay_data[b]):>6} samples, ratio={avg:.4f}, edge={((1-avg)*100):+.2f}%")
    
    print("\n" + "="*60)
    print("BACK (BSP $3-$8) - Best time to BACK")
    print("="*60)
    for b in sorted(time_buckets, reverse=True):
        if back_data[b]:
            avg = statistics.mean(back_data[b])
            print(f"  {b//60}m {b%60}s before: {len(back_data[b]):>6} samples, ratio={avg:.4f}, edge={((avg-1)*100):+.2f}%")

if __name__ == "__main__":
    main()
