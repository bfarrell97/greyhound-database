"""
Debug Import Matching
Check why API files aren't matching DB entries.
"""
import sqlite3
import requests
import bz2
import json
from datetime import datetime
from urllib.parse import quote

BASE_URL = "https://historicdata.betfair.com/api/"
SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0="

TRACK_MAPPING = {
    'RICHMOND (RIS)': 'RICHMOND STRAIGHT',
    'BET DELUXE CAPALABA': 'CAPALABA',
    'MEADOWS (MEP)': 'THE MEADOWS',
    'MURRAY BRIDGE (MBS)': 'MURRAY BRIDGE',
    'MURRAY BRIDGE (MBR)': 'MURRAY BRIDGE',
}

def normalize_name(name):
    if not name: return ''
    name = name.upper().strip()
    for old, new in TRACK_MAPPING.items():
        name = name.replace(old, new)
    return name

print("Building DB Lookup sample...")
conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()
cursor.execute("""
    SELECT t.TrackName, rm.MeetingDate, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate LIKE '2024-11-%'
    LIMIT 20
""")
db_keys = []
print("Sample DB Keys (Nov 2024):")
for track, date, name in cursor.fetchall():
    key = (normalize_name(track), date[:10], normalize_name(name))
    db_keys.append(key)
    print(f"  DB: {key}")
conn.close()

# Get one file from API
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": 1, "fromMonth": 11, "fromYear": 2024,
    "toDay": 1, "toMonth": 11, "toYear": 2024,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}
headers = {'content-type': 'application/json', 'ssoid': SSOID}
resp = requests.post(url, headers=headers, json=payload, timeout=30)
files = resp.json()

if files:
    print(f"\nChecking First API File: {files[0]}")
    encoded_path = quote(files[0], safe='')
    file_url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    r = requests.get(file_url, headers={'ssoid': SSOID})
    
    decompressed = bz2.decompress(r.content).decode('utf-8')
    lines = decompressed.strip().split('\n')
    data = json.loads(lines[-1])
    md = data['mc'][0]['marketDefinition']
    
    venue = normalize_name(md.get('venue', ''))
    market_time = md.get('marketTime', '')[:10]
    
    print(f"API Venue: {venue}")
    print(f"API Date: {market_time}")
    
    print("API Runners:")
    for runner in md.get('runners', []):
        name = normalize_name(runner.get('name', ''))
        key = (venue, market_time, name)
        print(f"  API: {key}")
else:
    print("No API files found for date")
