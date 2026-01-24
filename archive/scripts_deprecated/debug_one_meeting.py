"""Debug ONE specific meeting: Bet Deluxe Capalaba 2022-06-05"""
import sqlite3
import requests
import bz2
import json
import re
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

conn = sqlite3.connect('greyhound_racing.db')

print("="*70)
print("DEBUG: Bet Deluxe Capalaba 2022-06-05")
print("="*70)

# Get ALL DB entries for this track/date
q = """
SELECT g.GreyhoundName, ge.BSP
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Bet Deluxe Capalaba' AND rm.MeetingDate = '2022-06-05'
"""
db_entries = conn.execute(q).fetchall()
db_names_with_bsp = {normalize_name(row[0]) for row in db_entries if row[1]}
db_names_no_bsp = {normalize_name(row[0]) for row in db_entries if not row[1]}

print(f"\nDB Entries: {len(db_entries)}")
print(f"  With BSP: {len(db_names_with_bsp)}")
print(f"  Without BSP: {len(db_names_no_bsp)}")

print(f"\n  Sample WITH BSP: {list(db_names_with_bsp)[:5]}")
print(f"  Sample WITHOUT BSP: {list(db_names_no_bsp)[:5]}")

# Get API files for this date
dt = datetime(2022, 6, 5)
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": dt.day, "fromMonth": dt.month, "fromYear": dt.year,
    "toDay": dt.day, "toMonth": dt.month, "toYear": dt.year,
    "eventId": None, "eventName": None,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}
headers = {'content-type': 'application/json', 'ssoid': SSOID}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
files = resp.json() if resp.status_code == 200 else []
print(f"\nAPI Files: {len(files)}")

# Find CAPALABA files
capalaba_files = []
capalaba_venues = set()
bsp_dogs = {}

for fp in files:
    encoded_path = quote(fp, safe='')
    furl = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    try:
        fresp = requests.get(furl, headers={'ssoid': SSOID}, timeout=30)
        if fresp.status_code != 200:
            continue
        decompressed = bz2.decompress(fresp.content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        data = json.loads(lines[-1])
        if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
            md = data['mc'][0]['marketDefinition']
            venue = md.get('venue', '').upper()
            
            if 'CAPALABA' in venue:
                capalaba_files.append(fp)
                capalaba_venues.add(venue)
                for r in md.get('runners', []):
                    name = normalize_name(r.get('name', ''))
                    bsp = r.get('bsp')
                    if name and bsp and bsp > 0:
                        bsp_dogs[name] = bsp
    except Exception as e:
        continue

print(f"\nCAPALABA files found: {len(capalaba_files)}")
print(f"  Venues: {capalaba_venues}")
print(f"  BSP Dogs: {len(bsp_dogs)}")
print(f"  Sample: {list(bsp_dogs.items())[:5]}")

# Match analysis
matched = db_names_no_bsp & set(bsp_dogs.keys())
db_only = db_names_no_bsp - set(bsp_dogs.keys())
bsp_only = set(bsp_dogs.keys()) - (db_names_with_bsp | db_names_no_bsp)

print(f"\nMATCH ANALYSIS (for entries currently without BSP):")
print(f"  Should match (in both): {len(matched)}")
print(f"  DB only (no BSP match): {len(db_only)}")
print(f"  BSP only (no DB entry): {len(bsp_only)}")

if db_only:
    print(f"\n  DB names without BSP match:")
    for n in list(db_only)[:10]:
        print(f"    '{n}'")

if bsp_only:
    print(f"\n  BSP names without DB entry:")
    for n in list(bsp_only)[:10]:
        print(f"    '{n}'")

conn.close()
