import os
import sys
from datetime import datetime, timedelta, timezone

# Add root directory to path to allow importing src
sys.path.append(os.getcwd())
from src.integration.betfair_fetcher import BetfairOddsFetcher

def debug_sale():
    fetcher = BetfairOddsFetcher()
    fetcher.login()
    
    # Look back 12 hours
    from_time = datetime.utcnow() - timedelta(hours=12)
    to_time = datetime.utcnow() + timedelta(hours=12)
    
    print(f"Querying markets from {from_time} to {to_time}...")
    
    # Try getting ALL greyhound markets in that window
    markets = fetcher.get_greyhound_markets(from_time=from_time, to_time=to_time)
    
    print(f"Total markets found: {len(markets)}")
    for m in markets:
        if 'Sale' in m.event.name:
            print(f"MATCH: {m.event.name} - {m.market_name} (ID: {m.market_id})")
            
    fetcher.logout()

if __name__ == "__main__":
    debug_sale()
