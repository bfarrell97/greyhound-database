"""Quick test - one day only"""
import bz2, json, os

base = r'data\bsp\1\BASIC\2025\Jan\1'
print(f'Testing: {base}', flush=True)

if os.path.exists(base):
    events = os.listdir(base)
    print(f'Found {len(events)} events', flush=True)
    
    count = 0
    for event in events[:3]:
        event_path = os.path.join(base, event)
        if not os.path.isdir(event_path):
            continue
        for market in os.listdir(event_path)[:5]:
            if market.endswith('.bz2'):
                count += 1
                file_path = os.path.join(event_path, market)
                with bz2.open(file_path, 'rt') as f:
                    lines = f.readlines()
                    data = json.loads(lines[-1])
                    if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
                        md = data['mc'][0]['marketDefinition']
                        venue = md.get('venue', 'UNKNOWN')
                        print(f'{count}: {venue}', flush=True)
    print('Done!', flush=True)
else:
    print('Path not found!', flush=True)
