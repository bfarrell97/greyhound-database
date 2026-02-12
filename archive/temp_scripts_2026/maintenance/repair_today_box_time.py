import sqlite3
import pandas as pd
import re
import os
import sys
from datetime import datetime, timezone, timedelta

# Add root directory to path to allow importing src
sys.path.append(os.getcwd())
try:
    from src.integration.betfair_fetcher import BetfairOddsFetcher
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.integration.betfair_fetcher import BetfairOddsFetcher

DB_PATH = 'greyhound_racing.db'

def repair_db_v2():
    print("--- REPAIR DB V2: Aggressive Matching ---")
    
    fetcher = BetfairOddsFetcher()
    if not fetcher.login(): return

    # Look back 6 hours, look ahead 12 hours
    print("Fetching Betfair markets (-6h to +12h)...")
    from_time = datetime.utcnow() - timedelta(hours=6)
    to_time = datetime.utcnow() + timedelta(hours=12)
    markets = fetcher.get_greyhound_markets(from_time=from_time, to_time=to_time)
    
    if not markets:
        print("[INFO] No Betfair markets found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load ALL today's runners
    today_str = datetime.now().strftime('%Y-%m-%d')
    query = f"""
    SELECT 
        ge.EntryID, ge.RaceID, g.GreyhoundName, t.TrackName, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = '{today_str}'
    """
    db_runners = pd.read_sql_query(query, conn)
    db_runners['DogClean'] = db_runners['GreyhoundName'].str.upper().str.strip()
    # Fixed replacement for substrings
    db_runners['TrackClean'] = db_runners['TrackName'].str.upper().str.replace('THE ', '').str.replace('MT ', 'MOUNT ').str.strip()
    
    updates_box = []
    updates_time = []
    
    matched_tracks = set()

    for m in markets:
        raw_event = m.event.name.upper()
        # Extract track: e.g. "SALE (AUS) 26TH DEC" -> "SALE"
        clean_track = raw_event.split(' (')[0].strip()
        clean_track = clean_track.replace('THE ', '').replace('MT ', 'MOUNT ')
        
        matched_tracks.add(clean_track)
        
        # Local Time
        try:
            dt_utc = m.market_start_time.replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone()
            m_time_local = dt_local.strftime('%H:%M')
        except:
            m_time_local = m.market_start_time.strftime('%H:%M')

        # Race Number
        race_match = re.search(r'R(\d+)', m.market_name, re.I)
        race_num = int(race_match.group(1)) if race_match else None
        
        if race_num:
            r_ids = db_runners[(db_runners['TrackClean'] == clean_track) & (db_runners['RaceNumber'] == race_num)]['RaceID'].unique()
            for rid in r_ids:
                updates_time.append((m_time_local, int(rid)))

        # Runners
        for r in m.runners:
            name = re.sub(r'^\d+\.\s*', '', r.runner_name).strip().upper()
            
            box = 1
            try:
                if hasattr(r, 'metadata') and r.metadata and 'TRAP' in r.metadata:
                    box = int(r.metadata['TRAP'])
                else:
                    name_match = re.match(r'^(\d+)\.', r.runner_name)
                    if name_match: box = int(name_match.group(1))
            except: pass
            
            # Match
            mask = (db_runners['TrackClean'] == clean_track) & (db_runners['DogClean'] == name)
            entry_ids = db_runners[mask]['EntryID'].tolist()
            
            if entry_ids:
                for eid in entry_ids:
                    updates_box.append((int(box), int(eid)))

    print(f"Found {len(matched_tracks)} tracks in Betfair: {sorted(list(matched_tracks))}")
    print(f"Targeting {len(db_runners['TrackClean'].unique())} tracks in DB: {sorted(db_runners['TrackClean'].unique().tolist())}")

    cursor = conn.cursor()
    if updates_time:
        print(f"Applying {len(updates_time)} Race Time updates...")
        cursor.executemany("UPDATE Races SET RaceTime = ? WHERE RaceID = ?", list(set(updates_time)))
        
    if updates_box:
        print(f"Applying {len(updates_box)} Box Number updates...")
        cursor.executemany("UPDATE GreyhoundEntries SET Box = ? WHERE EntryID = ?", updates_box)
        
    conn.commit()
    conn.close()
    fetcher.logout()
    print("\n[OK] Repair V2 Complete.")

if __name__ == "__main__":
    repair_db_v2()
