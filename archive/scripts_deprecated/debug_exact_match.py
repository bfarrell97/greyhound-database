"""Debug matching - compare exact data for one specific date/track"""
import requests
import json
import sqlite3
import bz2
import re
from urllib.parse import quote

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

conn = sqlite3.connect(DB_PATH)

# Get ALL entries for Nov 1 2025 that need BSP
print("="*60)
print("DB entries needing BSP for 2025-11-01:")
print("="*60)
query = """
SELECT UPPER(t.TrackName), g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.BSP IS NULL AND rm.MeetingDate = '2025-11-01'
"""
db_entries = conn.execute(query).fetchall()
print(f"Total: {len(db_entries)} entries need BSP")
print("\nBy track:")
track_counts = {}
for track, name in db_entries:
    track_counts[track] = track_counts.get(track, 0) + 1
for track, count in sorted(track_counts.items()):
    print(f"  {track}: {count}")

# Build DB lookup
db_lookup = {}
for track, name in db_entries:
    normalized = normalize_name(name)
    key = (track, '2025-11-01', normalized)
    db_lookup[key] = name

# Now get BSP data for Nov 1
print("\n" + "="*60)
print("BSP API data for 2025-11-01:")
print("="*60)

from datetime import datetime
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": 1, "fromMonth": 11, "fromYear": 2025,
    "toDay": 1, "toMonth": 11, "toYear": 2025,
    "eventId": None, "eventName": None,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}
headers = {'content-type': 'application/json', 'ssoid': SSOID}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
files = resp.json() if resp.status_code == 200 else []

print(f"Total files: {len(files)}")

# Download all and extract data
bsp_entries = []
bsp_tracks = set()
for fp in files:
    encoded_path = quote(fp, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    resp = requests.get(url, headers={'ssoid': SSOID}, timeout=30)
    if resp.status_code != 200:
        continue
    try:
        decompressed = bz2.decompress(resp.content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        data = json.loads(lines[-1])
        if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
            md = data['mc'][0]['marketDefinition']
            venue = md.get('venue', '').upper()
            market_time = md.get('marketTime', '')[:10]
            bsp_tracks.add(venue)
            for r in md.get('runners', []):
                bsp = r.get('bsp')
                dog_name = normalize_name(r.get('name', ''))
                bsp_entries.append((venue, market_time, dog_name, bsp))
    except:
        pass

print(f"Total BSP entries: {len(bsp_entries)}")
print("\nBSP Venues:")
for v in sorted(bsp_tracks):
    print(f"  {v}")

# Compare
print("\n" + "="*60)
print("COMPARISON:")
print("="*60)

# Tracks in DB but not in BSP
db_tracks = set(track_counts.keys())
missing_tracks = db_tracks - bsp_tracks
print(f"\nDB tracks NOT in BSP data: {missing_tracks}")

# Tracks in BSP but not in DB
extra_tracks = bsp_tracks - db_tracks
print(f"BSP tracks NOT in DB: {extra_tracks}")

# For matching track, compare names
common_tracks = db_tracks & bsp_tracks
print(f"\nMatching tracks: {common_tracks}")

for track in list(common_tracks)[:2]:
    print(f"\n--- {track} ---")
    
    db_names = set()
    for t, n in db_entries:
        if t == track:
            db_names.add(normalize_name(n))
    
    bsp_names = set()
    for v, d, n, b in bsp_entries:
        if v == track and d == '2025-11-01':
            bsp_names.add(n)
    
    print(f"DB names: {len(db_names)}")
    print(f"BSP names: {len(bsp_names)}")
    
    matched = db_names & bsp_names
    db_only = db_names - bsp_names
    bsp_only = bsp_names - db_names
    
    print(f"Matched: {len(matched)}")
    print(f"DB only (not in BSP): {len(db_only)}")
    if db_only:
        print(f"  Examples: {list(db_only)[:5]}")
    print(f"BSP only (not in DB): {len(bsp_only)}")
    if bsp_only:
        print(f"  Examples: {list(bsp_only)[:5]}")

conn.close()
