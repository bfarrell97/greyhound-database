"""Debug dog name matching using local BSP files"""
import sqlite3
import bz2
import json
import re
import os

def normalize_name(name):
    """Our current normalization"""
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

conn = sqlite3.connect('greyhound_racing.db')

print("="*70)
print("DEBUG: Dog name matching using local BSP files")
print("="*70)

# Pick a track and date we should have data for
track_db = 'Bet Deluxe Capalaba'
track_bsp = 'CAPALABA'
test_date = '2025-09-10'  # Date with 78 entries needing BSP

# Get DB entries for this track/date
query = """
SELECT g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = ? AND rm.MeetingDate = ?
"""
db_dogs = conn.execute(query, (track_db, test_date)).fetchall()
db_names = set(normalize_name(d[0]) for d in db_dogs)

print(f"\nDB: {track_db} on {test_date}")
print(f"  {len(db_dogs)} entries, {len(db_names)} unique dogs")
for name in list(db_names)[:10]:
    print(f"    {name}")

# Get BSP entries from local files for same date
bsp_folder = r'data\bsp\1\BASIC\2025\Sep\10'
bsp_names = set()

if os.path.exists(bsp_folder):
    for event in os.listdir(bsp_folder):
        event_path = os.path.join(bsp_folder, event)
        if not os.path.isdir(event_path):
            continue
        for f in os.listdir(event_path):
            if not f.endswith('.bz2'):
                continue
            try:
                with bz2.open(os.path.join(event_path, f), 'rt') as fh:
                    lines = fh.readlines()
                data = json.loads(lines[-1])
                if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
                    md = data['mc'][0]['marketDefinition']
                    venue = md.get('venue', '').upper()
                    if track_bsp in venue:
                        for r in md.get('runners', []):
                            name = r.get('name', '')
                            bsp_names.add(normalize_name(name))
            except:
                pass

print(f"\nBSP: {track_bsp} on {test_date}")
print(f"  {len(bsp_names)} unique dogs")
for name in list(bsp_names)[:10]:
    print(f"    {name}")

# Compare
matched = db_names & bsp_names
db_only = db_names - bsp_names
bsp_only = bsp_names - db_names

print(f"\n" + "="*70)
print("COMPARISON:")
print(f"  Matched: {len(matched)}")
print(f"  DB only (not in BSP): {len(db_only)}")
print(f"  BSP only (not in DB): {len(bsp_only)}")

if db_only:
    print(f"\n  DB dogs NOT found in BSP:")
    for n in list(db_only)[:10]:
        print(f"    {n}")

if bsp_only:
    print(f"\n  BSP dogs NOT found in DB:")
    for n in list(bsp_only)[:10]:
        print(f"    {n}")

conn.close()
