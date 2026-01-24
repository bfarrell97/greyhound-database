"""Debug why BSP matching is failing for specific tracks"""
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
    name = re.sub(r"['\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

print("="*70)
print("DEBUG: Why BSP matching fails")
print("="*70)

conn = sqlite3.connect('greyhound_racing.db')

# Sample entries from low-coverage tracks that need BSP
for db_track, bsp_track in [
    ('Bet Deluxe Capalaba', 'CAPALABA'),
    ('Meadows (MEP)', 'THE MEADOWS'),
    ('Richmond (RIS)', 'RICHMOND STRAIGHT'),
]:
    print(f"\n--- {db_track} -> {bsp_track} ---")
    
    # Get a date where we have entries needing BSP
    q = f"""
    SELECT rm.MeetingDate, COUNT(*)
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE t.TrackName = '{db_track}' AND ge.BSP IS NULL
    AND rm.MeetingDate BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY rm.MeetingDate
    ORDER BY COUNT(*) DESC LIMIT 1
    """
    result = conn.execute(q).fetchone()
    if not result:
        print("  No entries needing BSP in 2024")
        continue
    
    test_date = result[0]
    print(f"  Test Date: {test_date}, {result[1]} entries need BSP")
    
    # Get DB dogs for this track/date
    q = f"""
    SELECT g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE t.TrackName = '{db_track}' AND rm.MeetingDate = '{test_date}'
    """
    db_dogs = set(normalize_name(row[0]) for row in conn.execute(q).fetchall())
    print(f"  DB Dogs: {len(db_dogs)}")
    print(f"    Sample: {list(db_dogs)[:5]}")
    
    # Get BSP from API for that date
    dt = datetime.strptime(test_date, "%Y-%m-%d")
    
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
    print(f"  API Files for {test_date}: {len(files)}")
    
    # Find files for our track
    bsp_dogs = set()
    venue_names = set()
    for fp in files[:50]:
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
                venue_names.add(venue)
                if bsp_track in venue or venue in bsp_track:
                    for r in md.get('runners', []):
                        name = r.get('name', '')
                        bsp_dogs.add(normalize_name(name))
        except:
            pass
    
    print(f"  Unique Venues in API: {venue_names}")
    print(f"  BSP Dogs for {bsp_track}: {len(bsp_dogs)}")
    if bsp_dogs:
        print(f"    Sample: {list(bsp_dogs)[:5]}")
    
    # Match
    matched = db_dogs & bsp_dogs
    db_only = db_dogs - bsp_dogs
    bsp_only = bsp_dogs - db_dogs
    
    print(f"  Matched: {len(matched)}")
    if db_only:
        print(f"  DB only (unmatched): {list(db_only)[:5]}")
    if bsp_only:
        print(f"  BSP only: {list(bsp_only)[:5]}")

conn.close()
