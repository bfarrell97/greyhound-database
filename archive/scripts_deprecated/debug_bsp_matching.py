"""Debug BSP matching - check exact matching logic"""
import requests
import json
import sqlite3
import bz2
import re
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'
DB_PATH = 'greyhound_racing.db'

def download_list_of_files(from_date, to_date):
    url = BASE_URL + 'DownloadListOfFiles'
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": from_date.day,
        "fromMonth": from_date.month,
        "fromYear": from_date.year,
        "toDay": to_date.day,
        "toMonth": to_date.month,
        "toYear": to_date.year,
        "eventId": None,
        "eventName": None,
        "marketTypesCollection": ["WIN"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    return resp.json() if resp.status_code == 200 else []

def download_file(file_path):
    encoded_path = quote(file_path, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    headers = {'ssoid': SSOID}
    resp = requests.get(url, headers=headers, timeout=60)
    return resp.content if resp.status_code == 200 else None

def extract_dog_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    return match.group(1).strip().upper() if match else name.upper()

print("="*60, flush=True)
print("DEBUG: Check exact matching logic", flush=True)
print("="*60, flush=True)

conn = sqlite3.connect(DB_PATH)

# Get entries for BALLARAT for September 2025
print("\nDB entries for BALLARAT in September 2025:", flush=True)
query = """
SELECT rm.MeetingDate, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE UPPER(t.TrackName) = 'BALLARAT'
  AND rm.MeetingDate >= '2025-09-01'
  AND rm.MeetingDate <= '2025-09-30'
LIMIT 20
"""
db_samples = conn.execute(query).fetchall()
print(f"Found {len(db_samples)} entries")
for date, name in db_samples[:10]:
    print(f"  {date} - {name}", flush=True)

# Get BSP files for BALLARAT
from_date = datetime(2025, 9, 1)
to_date = datetime(2025, 9, 30)

print(f"\nFetching BSP files for September 2025 (AU only)...", flush=True)
files = download_list_of_files(from_date, to_date)
print(f"Found {len(files)} files", flush=True)

# Find BALLARAT files
bsp_samples = []
for fp in files[:200]:
    content = download_file(fp)
    if content:
        try:
            decompressed = bz2.decompress(content).decode('utf-8')
            lines = decompressed.strip().split('\n')
            if lines:
                data = json.loads(lines[-1])
                if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
                    md = data['mc'][0]['marketDefinition']
                    venue = md.get('venue', '').upper()
                    if venue == 'BALLARAT':
                        market_time = md.get('marketTime', '')
                        for r in md.get('runners', []):
                            dog_name = extract_dog_name(r.get('name', ''))
                            bsp = r.get('bsp')
                            bsp_samples.append((market_time, dog_name, bsp))
        except:
            pass
    
    if len(bsp_samples) >= 20:
        break

print(f"\nBSP samples from BALLARAT:", flush=True)
for mtime, name, bsp in bsp_samples[:15]:
    print(f"  {mtime[:10]} - {name} (BSP: {bsp})", flush=True)

# Check for exact matches
print("\n\nChecking for exact matches...", flush=True)
for date, db_name in db_samples[:5]:
    for mtime, bsp_name, bsp in bsp_samples:
        if date == mtime[:10] and db_name.upper() == bsp_name:
            print(f"  MATCH: {date} - {db_name} = {bsp}", flush=True)
            break
    else:
        # No match found
        print(f"  NO MATCH: DB has {date} - {db_name}", flush=True)
        # Show close BSP entries
        for mtime, bsp_name, bsp in bsp_samples[:3]:
            if mtime[:10] == date:
                print(f"    BSP has: {mtime[:10]} - {bsp_name}", flush=True)

conn.close()
