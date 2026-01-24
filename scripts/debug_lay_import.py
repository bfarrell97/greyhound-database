"""Debug script to identify track name matching issues"""
import sqlite3
import requests
import bz2
import json
import re
from urllib.parse import quote

SSOID = 'Go3cWrYE2AaXgxNdU0Zs/HMWFmdAV9FAiPnI3wLJI6k='
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match: 
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-\.]", '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

# Build lookup
conn = sqlite3.connect(DB_PATH)
query = """
SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2024-11-01' AND rm.MeetingDate <= '2024-11-02'
"""
lookup = {}
for entry_id, track, date, dog_name in conn.execute(query).fetchall():
    bsp_track = TRACK_MAPPING.get(track, track)
    normalized = normalize_name(dog_name)
    lookup[(bsp_track, date, normalized)] = entry_id
conn.close()
print(f'Lookup has {len(lookup)} entries')

# Sample lookup keys
print('\nSample LOOKUP keys (DB-side):')
for i, k in enumerate(list(lookup.keys())[:5]):
    print(f'  {k}')

# Get one file and check its venue
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    'sport': 'Greyhound Racing', 'plan': 'Basic Plan',
    'fromDay': 1, 'fromMonth': 11, 'fromYear': 2024,
    'toDay': 2, 'toMonth': 11, 'toYear': 2024,
    'marketTypesCollection': ['WIN'], 'countriesCollection': ['AU'], 'fileTypeCollection': ['M']
}
resp = requests.post(url, headers={'content-type': 'application/json', 'ssoid': SSOID}, json=payload, timeout=30)
files = resp.json()
print(f'\nGot {len(files)} files from API')

if files:
    fp = files[0]
    encoded = quote(fp, safe='')
    resp = requests.get(f'{BASE_URL}DownloadFile?filePath={encoded}', headers={'ssoid': SSOID}, timeout=30)
    data = bz2.decompress(resp.content).decode('utf-8')
    lines = data.strip().split('\n')
    last = json.loads(lines[-1])
    md = last['mc'][0]['marketDefinition']
    
    venue = md.get('venue', '').upper()
    market_time_str = md.get('marketTime', '')[:10]
    
    print(f'\nFile venue: "{venue}"')
    print(f'File date: "{market_time_str}"')
    
    print('\nFile runners (keys to lookup):')
    for r in md.get('runners', [])[:3]:
        name = normalize_name(r.get('name', ''))
        key = (venue, market_time_str, name)
        found = key in lookup
        print(f'  {key} -> Found: {found}')

# Now check if the file has lay price data (atl = available to lay)
print('\n--- Checking for LAY price data in file ---')
from datetime import datetime
market_time = datetime.fromisoformat(md.get('marketTime', '').replace('Z', '+00:00'))

has_atl = False
sample_atl = None
for line in lines[:-1][:100]:  # Check first 100 data lines
    try:
        d = json.loads(line)
        if 'mc' in d and d['mc']:
            for rc in d['mc'][0].get('rc', []):
                atl = rc.get('atl', [])
                if atl:
                    has_atl = True
                    if sample_atl is None:
                        sample_atl = atl
                        pt = d.get('pt')
                        if pt:
                            pub_time = datetime.fromtimestamp(pt/1000, tz=market_time.tzinfo)
                            secs_before = (market_time - pub_time).total_seconds()
                            print(f'Sample ATL found: {atl} (at {secs_before:.0f} secs before market)')
    except:
        continue

print(f'\nHas any ATL data: {has_atl}')

# Also check what 'rc' data looks like
print('\n--- Sample runner change record ---')
for line in lines[10:15]:
    try:
        d = json.loads(line)
        if 'mc' in d and d['mc'] and d['mc'][0].get('rc'):
            rc = d['mc'][0]['rc'][0]
            print(f'rc keys: {list(rc.keys())}')
            break
    except:
        continue
