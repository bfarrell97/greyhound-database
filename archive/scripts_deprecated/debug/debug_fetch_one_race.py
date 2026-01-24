import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.betfair_fetcher import BetfairOddsFetcher

def debug_fetch_bendigo_r1():
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        return

    print("Fetching Bendigo Race 1...")
    # Get all markets to see if it's there
    markets = fetcher.get_greyhound_markets()
    
    found = False
    for m in markets:
        # Check name
        if "bendigo" in m.event.name.lower() or "bendigo" in m.market_name.lower():
            if "r1 " in m.market_name.lower() or "race 1" in m.market_name.lower():
                print(f"FOUND: {m.market_name} | ID: {m.market_id} | Time: {m.market_start_time}")
                found = True
                
                # Get Odds
                odds = fetcher.get_market_odds(m.market_id)
                print(f"Odds Count: {len(odds)}")
                print(f"Sample Odds: {list(odds.values())[:5]}")
                break
    
    if not found:
        print("Bendigo Race 1 NOT FOUND in API response.")
        print("Total Markets Returned:", len(markets))
        # Print first 5 to see what we are getting
        for m in markets[:5]:
            print(f"  > {m.market_name} ({m.market_start_time})")

    fetcher.logout()

if __name__ == "__main__":
    debug_fetch_bendigo_r1()
