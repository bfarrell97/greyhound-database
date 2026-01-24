"""
Inspect Betfair History File Structure
Check if files contain LTP (Last Traded Price) data
"""
import os
import bz2
import json

# Find a sample file
bsp_dir = 'data/bsp'

sample_file = None
for root, dirs, files in os.walk(bsp_dir):
    for f in files:
        if f.endswith('.bz2'):
            sample_file = os.path.join(root, f)
            break
    if sample_file:
        break

if not sample_file:
    print("No .bz2 files found")
    exit()

print(f"Inspecting: {sample_file}")
print("="*80)

with bz2.open(sample_file, 'rt', encoding='utf-8') as f:
    lines = f.readlines()

print(f"File has {len(lines)} lines (market updates)")

# Check first and last lines (they usually have different data)
print("\n--- FIRST LINE (Market Open) ---")
first_data = json.loads(lines[0])
print(json.dumps(first_data, indent=2)[:2000])

print("\n--- LAST LINE (Market Close/Result) ---")
last_data = json.loads(lines[-1])
print(json.dumps(last_data, indent=2)[:2000])

# Look for LTP specifically
print("\n--- SEARCHING FOR LTP DATA ---")
found_ltp = False
for i, line in enumerate(lines[-10:]):  # Check last 10 lines
    data = json.loads(line)
    if 'mc' in data:
        for mc in data['mc']:
            if 'rc' in mc:  # Runner Changes
                for rc in mc['rc']:
                    if 'ltp' in rc:
                        print(f"Found LTP in line {len(lines) - 10 + i}!")
                        print(f"  Runner ID: {rc.get('id')}, LTP: {rc.get('ltp')}")
                        found_ltp = True

if not found_ltp:
    print("No LTP data found in sample file.")
    print("\nChecking available keys in runner data...")
    for line in lines[-5:]:
        data = json.loads(line)
        if 'mc' in data:
            for mc in data['mc']:
                if 'rc' in mc:
                    for rc in mc['rc']:
                        print(f"Runner keys: {list(rc.keys())}")
                        break
