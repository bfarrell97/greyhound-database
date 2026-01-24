
import sqlite3
import os
import bz2
import json
import re
import sys
from datetime import datetime, timedelta

def check_db_range():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("\n--- DB DATE RANGE ---")
    cursor.execute("SELECT MIN(MeetingDate), MAX(MeetingDate), COUNT(*) FROM RaceMeetings")
    row = cursor.fetchone()
    print(f"Min Date: {row[0]}")
    print(f"Max Date: {row[1]}")
    print(f"Total Meetings: {row[2]}")
    
    conn.close()

def debug_file(file_path):
    print(f"\n--- DEBUGGING FILE: {file_path} ---")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    # Extract Data
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        
    last_line = lines[-1]
    data = json.loads(last_line)
    
    # Locate Def
    market_def = None
    if 'mc' in data:
        for mc in data['mc']:
            if 'marketDefinition' in mc:
                market_def = mc['marketDefinition']
                break
                
    if not market_def:
        print("No Market Definition found.")
        return

    # Extract metadata
    event_name = market_def.get('eventName', '')
    market_time_str = market_def.get('marketTime', '')
    
    print(f"Event Name: {repr(event_name)}")
    print(f"Market Time (UTC): {repr(market_time_str)}")
    print(f"Event Type ID: {repr(market_def.get('eventTypeId'))}")
    print(f"Market Type: {repr(market_def.get('marketType'))}")
    
    # Parse helpers
    track_match = re.search(r'^([^(]+)', event_name)
    track_name = track_match.group(1).strip() if track_match else event_name
    
    try:
        dt = datetime.strptime(market_time_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        race_date_utc = dt.strftime('%Y-%m-%d')
        # Local Time (Approx +10h for AUS)
        dt_local = dt + timedelta(hours=10) # Crude approx
        race_date_local = dt_local.strftime('%Y-%m-%d')
    except:
        print("Date parse failed.")
        race_date_utc = "ERROR"
        race_date_local = "ERROR"

    print(f"Derived Track: {repr(track_name)}")
    print(f"Derived Date (UTC): {race_date_utc}")
    print(f"Derived Date (Local?): {race_date_local}")

    # DB Checks
    sys.stdout.flush()
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check Track
    print("\n[CHECK 1] Track Match:")
    cursor.execute("SELECT TrackID, TrackName FROM Tracks WHERE TrackName LIKE ?", (f"{track_name}%",))
    tracks = cursor.fetchall()
    if tracks:
        print(f"  Found Tracks: {tracks}")
    else:
        print(f"  NO TRACK FOUND matching '{track_name}%'")
        cursor.execute("SELECT TrackName FROM Tracks LIMIT 5")
        print(f"  Sample Tracks: {cursor.fetchall()}")

    # Check Meeting (UTC vs Local)
    print("\n[CHECK 2] Meeting Match:")
    for d in [race_date_utc, race_date_local]:
        print(f"  Checking {d}...")
        cursor.execute("""
            SELECT m.MeetingID, t.TrackName 
            FROM RaceMeetings m 
            JOIN Tracks t ON m.TrackID = t.TrackID
            WHERE m.MeetingDate = ? AND t.TrackName LIKE ?
        """, (d, f"{track_name}%"))
        mtgs = cursor.fetchall()
        if mtgs:
            print(f"    FOUND Meeting: {mtgs}")
        else:
            print(f"    No meeting found for {d}")

    # Check Dog Names
    print("\n[CHECK 3] Dog Match (Sample):")
    if 'runners' in market_def:
        runner = market_def['runners'][0]
        name = runner.get('name', '')
        print(f"  Raw Runner Name: {name}")
        
        # Clean
        clean_name_match = re.search(r'^\d+\.\s+(.*)', name)
        clean_name = clean_name_match.group(1) if clean_name_match else name
        print(f"  Clean Runner Name: {clean_name}")
        
        # Exact Match
        cursor.execute("SELECT GreyhoundID, GreyhoundName FROM Greyhounds WHERE GreyhoundName = ?", (clean_name,))
        exact = cursor.fetchall()
        if exact:
             print(f"    EXACT Match found: {exact}")
        else:
             print(f"    NO Exact match.")
             # Case insensitive
             cursor.execute("SELECT GreyhoundID, GreyhoundName FROM Greyhounds WHERE GreyhoundName = ? COLLATE NOCASE", (clean_name,))
             nocase = cursor.fetchall()
             if nocase:
                 print(f"    CASE-INSENSITIVE Match found: {nocase}")
             else:
                 print("    Attributes match failed.")

    conn.close()

# Redirect stdout to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("debug_output.txt", "w", encoding='utf-8')

    def write(self, message):
        # self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

if __name__ == "__main__":
    check_db_range()
    # Pick a real file if possible
    debug_file(r'data/bsp/BASIC/2025/Apr/1/34175726/1.241667158.bz2')
