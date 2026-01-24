import sys
import os
import requests
import sqlite3
import pandas as pd
from datetime import datetime

# Mock the API fetch for debugging
def get_topaz_runs(state, year, month):
    # Just fetch VIC for Dec 2025 (Sale is in VIC)
    url = f"https://api.topaz.racing/v1/greyhound/runs/{state}/{year}/{month}"
    # Note: User might not have this URL valid without headers, but let's try or mock it.
    # Actually, import_topaz_history uses TopazAPI class. I should use that.
    pass

sys.path.append(os.getcwd())
from src.integration.topaz_api import TopazAPI

def debug_holly():
    api = TopazAPI()
    
    # 1. Get DB Entry
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        g.GreyhoundName, 
        r.RaceNumber,
        rm.MeetingDate,
        t.TrackName,
        ge.EntryID
    FROM GreyhoundEntries ge 
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID 
    JOIN Races r ON ge.RaceID = r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
    JOIN Tracks t ON rm.TrackID = t.TrackID 
    WHERE g.GreyhoundName LIKE '%HOLLY ROSE%' 
    AND rm.MeetingDate = '2025-12-26'
    """
    db_row = pd.read_sql_query(query, conn).iloc[0]
    conn.close()
    
    print("=== DB CONTEXT ===")
    print(db_row)
    
    db_key = (db_row['MeetingDate'], int(db_row['RaceNumber']), db_row['GreyhoundName'].replace(' ', '').upper())
    print(f"DB Matching Key: {db_key}")
    
    # 2. Fetch Topaz Data
    print("\n=== TOPAZ DATA ===")
    # Sale is VIC.
    runs = api.get_bulk_runs_by_month('VIC', 2025, 12)
    
    found_any = False
    for run in runs:
        # Check if it looks like Holly
        r_dog = run.get('dogName', '').upper()
        if 'HOLLY' in r_dog and 'ROSE' in r_dog:
            found_any = True
            r_date = run.get('meetingDate', '').split('T')[0]
            r_num = run.get('raceNumber')
            r_tr = run.get('trackName', 'Unknown')
            
            # Key used in script
            r_dog_norm = r_dog.replace(' ', '').replace("'", "").replace(".", "")
            
            print(f"Found Candidate: {r_dog}")
            print(f"  Date: {r_date} (DB wants {db_row['MeetingDate']})")
            print(f"  Race: {r_num} (DB wants {db_row['RaceNumber']})")
            print(f"  Track: {r_tr}")
            print(f"  Generated Key: {(r_date, r_num, r_dog_norm)}")
            
            if (r_date, r_num, r_dog_norm) == db_key:
                print("  MATCH: YES")
            else:
                print("  MATCH: NO")

    if not found_any:
        print("Topaz returned NO DOGS named 'HOLLY ROSE' for VIC in Dec 2025.")

if __name__ == "__main__":
    debug_holly()
