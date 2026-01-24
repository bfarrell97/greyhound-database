"""
Analyze best betting and laying times - 2024 & 2025 WIN MARKETS
With better progress output
"""
import bz2
import json
import os
import statistics
from datetime import datetime
from collections import defaultdict
import sys

def analyze_file(file_path):
    """Analyze price evolution in a single WIN market file"""
    results = []
    
    try:
        with bz2.open(file_path, 'rt') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return results
        
        last_data = json.loads(lines[-1])
        if 'mc' not in last_data or not last_data['mc']:
            return results
        
        mc = last_data['mc'][0]
        if 'marketDefinition' not in mc:
            return results
        
        md = mc['marketDefinition']
        
        # FILTER: Only WIN markets
        if md.get('marketType', '') != 'WIN':
            return results
        
        market_time_str = md.get('marketTime', '')
        if not market_time_str:
            return results
        
        try:
            market_time = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
        except:
            return results
        
        runner_bsp = {}
        for r in md.get('runners', []):
            bsp = r.get('bsp')
            if bsp and bsp > 0:
                runner_bsp[r.get('id')] = bsp
        
        if not runner_bsp:
            return results
        
        runner_prices = {sel_id: [] for sel_id in runner_bsp}
        
        for line in lines:
            try:
                data = json.loads(line)
                pt = data.get('pt')
                if not pt:
                    continue
                
                publish_time = datetime.fromtimestamp(pt / 1000, tz=market_time.tzinfo)
                seconds_to_start = (market_time - publish_time).total_seconds()
                
                if 'mc' in data and data['mc']:
                    rc = data['mc'][0].get('rc', [])
                    for r in rc:
                        sel_id = r.get('id')
                        ltp = r.get('ltp')
                        if sel_id and ltp and sel_id in runner_bsp:
                            runner_prices[sel_id].append((seconds_to_start, ltp))
            except:
                continue
        
        for sel_id, bsp in runner_bsp.items():
            prices = sorted(runner_prices.get(sel_id, []), key=lambda x: x[0], reverse=True)
            if prices:
                results.append({'bsp': bsp, 'prices': prices})
        
        return results
    except:
        return results

def main():
    print("="*70)
    print("BETTING TIMING ANALYSIS - 2024 & 2025 WIN MARKETS")
    print("="*70)
    sys.stdout.flush()
    
    lay_data = defaultdict(list)
    back_data = defaultdict(list)
    
    time_buckets = [600, 300, 120, 60, 30, 10, 5]
    
    files_analyzed = 0
    win_markets = 0
    
    # Folders with 2025 and 2024 data 
    folders = [
        (r'data\bsp\1\BASIC\2025', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']),
        (r'data\bsp\2\BASIC\2024', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
        (r'data\bsp\2\BASIC\2025', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']),
    ]
    
    for base_folder, months in folders:
        if not os.path.exists(base_folder):
            print(f"Skipping {base_folder} (not found)")
            sys.stdout.flush()
            continue
        
        print(f"\n{base_folder}:")
        sys.stdout.flush()
        
        for month in months:
            month_path = os.path.join(base_folder, month)
            if not os.path.exists(month_path):
                continue
            
            month_wins = 0
            
            for day in os.listdir(month_path):
                day_path = os.path.join(month_path, day)
                if not os.path.isdir(day_path):
                    continue
                
                for event in os.listdir(day_path):
                    event_path = os.path.join(day_path, event)
                    if not os.path.isdir(event_path):
                        continue
                    
                    for market_file in os.listdir(event_path):
                        if not market_file.endswith('.bz2'):
                            continue
                        
                        file_path = os.path.join(event_path, market_file)
                        results = analyze_file(file_path)
                        files_analyzed += 1
                        
                        if results:
                            win_markets += 1
                            month_wins += 1
                        
                        for r in results:
                            bsp = r['bsp']
                            prices = r['prices']
                            
                            for bucket in time_buckets:
                                for sec, price in prices:
                                    if abs(sec - bucket) < 15:
                                        ratio = price / bsp
                                        if bsp < 2.25:
                                            lay_data[bucket].append(ratio)
                                        elif 3 <= bsp <= 8:
                                            back_data[bucket].append(ratio)
                                        break
            
            print(f"  {month}: {month_wins:,} WIN markets")
            sys.stdout.flush()
    
    print(f"\nTotal: {files_analyzed:,} files, {win_markets:,} WIN markets")
    sys.stdout.flush()
    
    print("\n" + "="*70)
    print("LAY STRATEGY (BSP < $2.25): Best time to LAY")
    print("="*70)
    print(f"{'Time Before':<20} {'Samples':>10} {'Avg Ratio':>12} {'Lay Edge':>12}")
    print("-"*55)
    
    for bucket in sorted(time_buckets, reverse=True):
        if lay_data[bucket]:
            avg = statistics.mean(lay_data[bucket])
            edge = (1 - avg) * 100
            print(f"{bucket//60}m {bucket%60}s before{'':<8} {len(lay_data[bucket]):>10} {avg:>12.4f} {edge:>+11.2f}%")
    
    print("\n" + "="*70)
    print("BACK STRATEGY (BSP $3-$8): Best time to BACK")
    print("="*70)
    print(f"{'Time Before':<20} {'Samples':>10} {'Avg Ratio':>12} {'Back Edge':>12}")
    print("-"*55)
    
    for bucket in sorted(time_buckets, reverse=True):
        if back_data[bucket]:
            avg = statistics.mean(back_data[bucket])
            edge = (avg - 1) * 100
            print(f"{bucket//60}m {bucket%60}s before{'':<8} {len(back_data[bucket]):>10} {avg:>12.4f} {edge:>+11.2f}%")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("- Lay Edge > 0%: Odds are lower than BSP, good for laying")
    print("- Back Edge > 0%: Odds are higher than BSP, good for backing")
    print("="*70)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
