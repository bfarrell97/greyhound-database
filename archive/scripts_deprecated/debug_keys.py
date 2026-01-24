"""Debug: Show exact lookup key format"""
import sqlite3
import re

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

# Show sample keys that would be created
query = """
SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'Bet Deluxe Capalaba' AND rm.MeetingDate = '2022-06-05'
AND ge.BSP IS NULL
LIMIT 5
"""

print("LOOKUP KEYS (from DB):")
for entry_id, track, date, dog_name in conn.execute(query).fetchall():
    bsp_track = TRACK_MAPPING.get(track, track)
    normalized = normalize_name(dog_name)
    key = (bsp_track, date, normalized)
    print(f"  Key: {key}")
    print(f"    EntryID: {entry_id}")
    print(f"    Original Track: '{track}'")
    print(f"    Mapped Track: '{bsp_track}'")
    print()

conn.close()
