import sqlite3
import pandas as pd
import re
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add root directory to path to allow importing src
sys.path.append(os.getcwd())
try:
    from src.integration.topaz_api import TopazAPI
    from src.core.config import TOPAZ_API_KEY
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.integration.topaz_api import TopazAPI
    from src.core.config import TOPAZ_API_KEY

DB_PATH = 'greyhound_racing.db'

def repair_today_super_turbo():
    print("--- SUPER TURBO REPAIR: Box & Time (Source: Topaz) ---")
    
    api = TopazAPI(TOPAZ_API_KEY)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. FIND DELTA: Only tracks/races that have Box 0
    query = f"""
    SELECT DISTINCT t.TrackName, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = '{today_str}'
    AND (ge.Box = 0 OR ge.Box IS NULL)
    """
    delta = pd.read_sql_query(query, conn)
    if delta.empty:
        print("[OK] No runners with Box 0 found for today.")
        conn.close()
        return

    print(f"Found {len(delta)} races needing repair.")

    # 2. Get ALL meetings for today ONCE (Efficient)
    print("Fetching all national meetings...")
    states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']
    meeting_map = {} # (track_name_clean, race_num) -> meeting_id
    
    def fetch_meetings(state):
        try: return api.get_meetings(today_str, owning_authority_code=state)
        except: return []

    all_meetings = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = [executor.submit(fetch_meetings, s) for s in states]
        for f in as_completed(futures):
            all_meetings.extend(f.result())

    # Map TrackName -> meetingId
    track_to_mid = {}
    for m in all_meetings:
        t_name = m.get('trackName', '').upper().replace('THE ', '').replace('MT ', 'MOUNT ').strip()
        track_to_mid[t_name] = m.get('meetingId')

    # 3. Load DB runners for matching
    query_runners = f"""
    SELECT ge.EntryID, g.GreyhoundName, t.TrackName, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = '{today_str}'
    """
    db_runners = pd.read_sql_query(query_runners, conn)
    db_runners['DogClean'] = db_runners['GreyhoundName'].str.upper().str.strip()
    db_runners['TrackClean'] = db_runners['TrackName'].str.upper().str.replace('THE ', '').str.replace('MT ', 'MOUNT ').str.strip()

    # 4. Fetch Races for the relevant meetings
    updates_box = []
    m_ids_to_fetch = set()
    for _, row in delta.iterrows():
        t_clean = row['TrackName'].upper().replace('THE ', '').replace('MT ', 'MOUNT ').strip()
        mid = track_to_mid.get(t_clean)
        if mid: m_ids_to_fetch.add(mid)

    print(f"Fetching race data for {len(m_ids_to_fetch)} meetings...")
    all_race_data = [] # List of race objects
    
    def fetch_races(mid):
        try: return api.get_races(mid)
        except: return []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_races, mid) for mid in m_ids_to_fetch]
        for f in as_completed(futures):
            all_race_data.extend(f.result())

    # 5. MATCH & COLLECT
    print(f"Propagating boxes to {len(db_runners)} runners...")
    for race in all_race_data:
        r_num = race.get('raceNumber')
        t_name = race.get('trackName', '').upper().replace('THE ', '').replace('MT ', 'MOUNT ').strip()
        
        for runner in race.get('runners', []):
            d_name = runner.get('greyhoundName', '').upper().strip()
            box = runner.get('boxNumber', 0)
            
            if box > 0:
                mask = (db_runners['TrackClean'] == t_name) & (db_runners['RaceNumber'] == r_num) & (db_runners['DogClean'] == d_name)
                eids = db_runners[mask]['EntryID'].tolist()
                for eid in eids:
                    updates_box.append((int(box), int(eid)))

    if updates_box:
        print(f"Applying {len(updates_box)} Box Number updates...")
        cursor = conn.cursor()
        cursor.executemany("UPDATE GreyhoundEntries SET Box = ? WHERE EntryID = ?", updates_box)
        conn.commit()
    
    conn.close()
    print("[OK] Super Turbo Repair Complete.")

if __name__ == "__main__":
    repair_today_super_turbo()
