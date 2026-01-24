import sys
import os
import re
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.betfair_fetcher import BetfairOddsFetcher
from src.core.database import GreyhoundDatabase

def fix_bendigo():
    print("Initializing Fix Script for Bendigo R1...")
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Login failed")
        return

    db = GreyhoundDatabase()
    
    # Get markets
    print("Fetching markets...")
    markets = fetcher.get_greyhound_markets()
    
    found = False
    for m in markets:
        # Target Bendigo ALL
        if "bendigo" in m.event.venue.lower():
            print(f"FOUND: {m.market_name} at {m.event.venue}")
            found = True
            
            # Prepare Data
            race_num_match = re.search(r'R(\d+)', m.market_name)
            if not race_num_match:
                 race_num_match = re.search(r'Race\s+(\d+)', m.market_name)
            race_number = int(race_num_match.group(1)) if race_num_match else 0
            
            if race_number == 0: continue

            track_name = m.event.venue
            race_date_str = m.market_start_time.strftime('%Y-%m-%d')
            
            # Get Odds
            # print(f"Fetching Odds for R{race_number}...")
            odds = fetcher.get_market_odds(m.market_id)
            
            entries = []
            for runner in m.runners:
                clean_name = re.sub(r'^\d+\.\s*', '', runner.runner_name)
                price = odds.get(runner.selection_id)
                # print(f"  > {clean_name}: {price}")
                
                # extract box
                box = None
                prefix_match = re.match(r'^(\d+)\.', runner.runner_name)
                if prefix_match:
                    box = int(prefix_match.group(1))

                # Logic from predict_lay_strategy.py
                entries.append({
                    'greyhound_name': clean_name,
                    'box': box, 
                    'trainer': '',
                    'starting_price': price
                })
                
            race_data = {
                'track_name': track_name,
                'date': race_date_str,
                'race_number': race_number,
                'entries': entries
            }
            
            # THE FIX: Call with 3 arguments
            try:
                success = db.import_form_guide_data(race_data, race_date_str, track_name)
                # print(f"Import R{race_number} Success: {success}")
                if success:
                    print(f"[OK] R{race_number} Imported.")
                else:
                    print(f"[FAIL] R{race_number} Import Failed.")
            except Exception as e:
                print(f"Import Error R{race_number}: {e}")
                import traceback
                traceback.print_exc()
            
    if not found:
        print("Bendigo R1 not found in API.")
        
    fetcher.logout()
    
    # Verify DB
    print("\n--- DB VERIFICATION ---")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN StartingPrice IS NOT NULL THEN 1 ELSE 0 END)
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID=r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID
        JOIN Tracks t ON rm.TrackID=t.TrackID
        WHERE t.TrackName='Bendigo' AND r.RaceNumber=1 AND rm.MeetingDate='2025-12-13'
    """)
    row = cursor.fetchone()
    print(f"Entries: {row[0]}, WithOdds: {row[1]}")
    db.close()

if __name__ == "__main__":
    fix_bendigo()
