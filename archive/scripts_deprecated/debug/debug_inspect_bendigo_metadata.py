import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.betfair_fetcher import BetfairOddsFetcher

def inspect_metadata():
    print("Connecting to Betfair to inspect metadata...")
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        return

    markets = fetcher.get_greyhound_markets()
    for m in markets:
        if "bendigo" in m.event.venue.lower():
            print(f"\n--- MARKET: {m.market_name} ---")
            for runner in m.runners:
                print(f"Runner: {runner.runner_name}")
                print(f"  SelectionID: {runner.selection_id}")
                print(f"  Metadata: {runner.metadata}")
                # print(f"  Handicap: {runner.handicap}")
                # print(f"  SortPriority: {runner.sort_priority}")
                print("-" * 20)
            break
            
    fetcher.logout()

if __name__ == "__main__":
    inspect_metadata()
