"""
Check data/bsp/1 folder for SP data format
"""
import bz2
import json
import os

# Check folder 1 structure
base_path = r'c:\Users\bfarr\Dropbox\Python\greyhound-database\data\bsp\1'

print(f"Exploring: {base_path}")

# Find first valid file
for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith('.bz2'):
            file_path = os.path.join(root, f)
            print(f"\nFirst file found: {file_path}")
            
            try:
                with bz2.open(file_path, 'rt') as fp:
                    first_line = fp.readline()
                    data = json.loads(first_line)
                    
                    print(f"Keys: {data.keys()}")
                    
                    if 'mc' in data and data['mc']:
                        mc = data['mc'][0]
                        print(f"MC keys: {mc.keys()}")
                        
                        if 'marketDefinition' in mc:
                            md = mc['marketDefinition']
                            print(f"Venue: {md.get('venue')}")
                            print(f"Name: {md.get('name')}")
                            print(f"MarketTime: {md.get('marketTime')}")
                            print(f"Status: {md.get('status')}")
                            print(f"SettledTime: {md.get('settledTime')}")
                            
                            runners = md.get('runners', [])
                            print(f"\nRunners ({len(runners)}):")
                            for r in runners[:8]:
                                print(f"  {r.get('name')}: BSP={r.get('bsp')}, Status={r.get('status')}")
                    
                    # Also read last line for final data
                    fp.seek(0)
                    lines = fp.readlines()
                    print(f"\nTotal lines: {len(lines)}")
                    
                    if len(lines) > 1:
                        last_data = json.loads(lines[-1])
                        if 'mc' in last_data and last_data['mc']:
                            lmc = last_data['mc'][0]
                            if 'marketDefinition' in lmc:
                                lmd = lmc['marketDefinition']
                                lrunners = lmd.get('runners', [])
                                print("\nLast line runners (should have BSP):")
                                for r in lrunners[:8]:
                                    print(f"  {r.get('name')}: BSP={r.get('bsp')}")
                            
            except Exception as e:
                print(f"Error: {e}")
            
            # Only process first file
            break
    else:
        continue
    break
