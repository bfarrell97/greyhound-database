import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import datetime
from src.integration.betfair_fetcher import BetfairOddsFetcher
from src.core.database import GreyhoundDatabase
import datetime

def debug_persistence_loop():
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        return
        
    db = GreyhoundDatabase()
    
    print("Fetching markets...")
    markets = fetcher.get_greyhound_markets()
    print(f"Total Markets: {len(markets)}")
    
    for m in markets:
        # Check for Bendigo R1
        if "bendigo" in m.event.venue.lower() and ("r1 " in m.market_name.lower() or "race 1 " in m.market_name.lower()):
            print(f"\n--- PROCESSING BENDIGO R1 ---")
            print(f"Market ID: {m.market_id}")
            print(f"Venue: {m.event.venue}")
            print(f"Name: {m.market_name}")
            
            # 1. Check Race Number Extraction
            race_num_match = re.search(r'R(\d+)', m.market_name)
            if not race_num_match:
                race_num_match = re.search(r'Race\s+(\d+)', m.market_name)
            race_number = int(race_num_match.group(1)) if race_num_match else 0
            print(f"Extracted Race Num: {race_number}")
            
            # 2. Check DB Race ID
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT r.RaceID 
                FROM Races r
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                WHERE t.TrackName = ? AND r.RaceNumber = ? AND rm.MeetingDate = ?
            """, (m.event.venue, race_number, "2025-12-13"))
            row = cursor.fetchone()
            race_id = row[0] if row else None
            
            print(f"DB Race ID for '{m.event.venue}' R{race_number}: {race_id}")
            
            if not race_id:
                # Try partial match?
                print("Trying simplified track name 'Bendigo'...")
                cursor.execute("""
                    SELECT r.RaceID 
                    FROM Races r
                    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                    JOIN Tracks t ON rm.TrackID = t.TrackID
                    WHERE t.TrackName = ? AND r.RaceNumber = ? AND rm.MeetingDate = ?
                """, ("Bendigo", race_number, "2025-12-13"))
                row = cursor.fetchone()
                race_id = row[0] if row else None
                print(f"DB Race ID for 'Bendigo' R{race_number}: {race_id}")

            # 3. Simulate Import
            print("Fetching Odds...")
            odds = fetcher.get_market_odds(m.market_id)
            print(f"Odds Count: {len(odds)}")
            
            print("Runners:")
            for runner in m.runners:
                clean_name = re.sub(r'^\d+\.\s*', '', runner.runner_name)
                price = odds.get(runner.selection_id)
                print(f"  > {clean_name} (ID: {runner.selection_id}) Price: {price}")
                
                if race_id:
                    # Check if Runner Exists in DB
                    # We need GreyhoundID first
                    g_id = db.add_or_get_greyhound(clean_name)
                    print(f"    DB GreyhoundID: {g_id}")
                    
                    # Check Entry
                    cursor = db.conn.cursor()
                    cursor.execute("SELECT EntryID, StartingPrice FROM GreyhoundEntries WHERE RaceID=? AND GreyhoundID=?", (race_id, g_id))
                    row = cursor.fetchone()
                    print(f"    Existing Entry: {row}")
            
            break
            
    fetcher.logout()
    db.close()

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    debug_persistence_loop()
