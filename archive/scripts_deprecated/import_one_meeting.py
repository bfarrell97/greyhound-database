"""Import BSP for ONE SPECIFIC MEETING to test"""
import sqlite3
import requests
import bz2
import json
import re
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'

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
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

conn = sqlite3.connect('greyhound_racing.db')

# Build lookup for Capalaba 2022-06-05
print("Building lookup for Bet Deluxe Capalaba 2022-06-05...")
query = """
SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Bet Deluxe Capalaba' 
  AND rm.MeetingDate = '2022-06-05'
  AND ge.BSP IS NULL
"""
lookup = {}
for entry_id, track, date, dog_name in conn.execute(query).fetchall():
    bsp_track = TRACK_MAPPING.get(track, track)
    normalized = normalize_name(dog_name)
    lookup[(bsp_track, date, normalized)] = entry_id

print(f"Entries needing BSP: {len(lookup)}")

# Get API files for this day
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
print(f"API files for day: {len(files)}")

# Process ALL files
updates = []
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
        if 'mc' not in data or not data['mc']:
            continue
        if 'marketDefinition' not in data['mc'][0]:
            continue
        
        md = data['mc'][0]['marketDefinition']
        venue = md.get('venue', '').upper()
        market_time_str = md.get('marketTime', '')[:10]
        
        if 'CAPALABA' not in venue:
            continue
        
        for r in md.get('runners', []):
            name = normalize_name(r.get('name', ''))
            bsp = r.get('bsp')
            if name and bsp and bsp > 0:
                key = (venue, market_time_str, name)
                if key in lookup:
                    updates.append((bsp, lookup[key]))
    except:
        continue

print(f"\nMatches found: {len(updates)}")

if updates:
    print("Updating database...")
    conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", updates)
    conn.commit()
    print(f"Updated {len(updates)} entries!")

# Verify
cursor = conn.execute("""
SELECT COUNT(*), SUM(CASE WHEN BSP IS NOT NULL THEN 1 ELSE 0 END)
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Bet Deluxe Capalaba' AND rm.MeetingDate = '2022-06-05'
""")
total, with_bsp = cursor.fetchone()
print(f"\nFinal: {with_bsp}/{total} ({with_bsp/total*100:.1f}%) have BSP")

conn.close()
