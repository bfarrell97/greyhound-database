
import bz2
import json
import os

# Sample file path
file_path = r'data/bsp/BASIC/2025/Apr/1/34175726/1.241667158.bz2'

if not os.path.exists(file_path):
    # Try to find any bz2 file if that one is missing
    print("Sample file not found, searching...")
    for root, dirs, files in os.walk('data/bsp'):
        for f in files:
            if f.endswith('.bz2'):
                file_path = os.path.join(root, f)
                break
        if file_path != r'data/bsp/BASIC/2025/Apr/1/34175726/1.241667158.bz2':
            break

print(f"Probing: {file_path}")

try:
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Total Lines: {len(lines)}")
    
    # Print First Line (Market Def / Catalogue)
    print("\n--- FIRST LINE ---")
    print(lines[0][:500] + "...")
    
    # Print Last Line (Result / Closed)
    print("\n--- LAST LINE ---")
    print(lines[-1][:500] + "...")
    
    # Parse LAST line fully to find BSP in market definition
    print("\n--- LAST LINE PARSED ---")
    last_data = json.loads(lines[-1])
    # print(json.dumps(last_data, indent=2))
    
    # Check 'mc' list[0]['marketDefinition']['runners'] ?
    # Typically stream format: [{"op":"mcm", "mc": [{"marketDefinition": {"runners": [...]}}]}]
    
    if 'mc' in last_data:
        for mc in last_data['mc']:
            if 'marketDefinition' in mc:
                print("Found MarketDefinition in last line.")
                md = mc['marketDefinition']
                if 'runners' in md:
                    for runner in md['runners']:
                        # print(f"Runner: {runner.get('id')} BSP: {runner.get('bsp')}")
                        print(f"Runner: {runner}")
            elif 'rc' in mc:
                print("Found Runner Changes in last line.")
                for rc in mc['rc']:
                    print(f"RC: {rc}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
