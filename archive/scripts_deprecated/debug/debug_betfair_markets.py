import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.integration.betfair_fetcher import BetfairOddsFetcher
from datetime import datetime, timedelta

def debug_markets():
    print("Initializing Betfair Fetcher...")
    fetcher = BetfairOddsFetcher()
    
    if not fetcher.login():
        print("Login Failed")
        return

    print("\nFetching ALL AU Greyhound Markets (Next 24h)...")
    now = datetime.now()
    # Broaden window to ensure we catch everything
    markets = fetcher.get_greyhound_markets(
        from_time=now - timedelta(hours=12), # Look back a bit too just in case
        to_time=now + timedelta(hours=36)
    )
    
    print(f"Found {len(markets)} markets.")
    
    # Group by Track
    tracks = {}
    for m in markets:
        event_name = m.event.name
        if event_name not in tracks:
            tracks[event_name] = 0
            # Print first example to see naming format
            if "Taree" in event_name or "Angle" in event_name:
                print(f"  MATCH: {event_name} -> {m.market_name} ({m.market_id})")
        tracks[event_name] += 1
        
    print("\nSummary of Tracks Found:")
    for t in sorted(tracks.keys()):
        print(f"{t}: {tracks[t]} races")
        
    fetcher.logout()

if __name__ == "__main__":
    debug_markets()
