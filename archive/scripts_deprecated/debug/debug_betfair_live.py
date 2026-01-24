import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.betfair_fetcher import BetfairOddsFetcher
from datetime import datetime
import json

def debug_live_market():
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Login failed")
        return

    print("Fetching markets...")
    markets = fetcher.get_greyhound_markets()
    
    if not markets:
        print("No markets found.")
        return

    # Pick the next market to jump
    market = markets[0]
    print(f"\ninspecting Market: {market.market_name} ({market.market_id})")
    print(f"Start Time: {market.market_start_time}")
    
    # Get Odds
    print("\nFetching Market Book...")
    odds = fetcher.get_market_odds(market.market_id)
    print(f"Odds Map Keys: {list(odds.keys())}")
    
    # Manual inspection of book to see RAW values
    from betfairlightweight import filters
    book = fetcher.trading.betting.list_market_book(
        market_ids=[market.market_id],
        price_projection=filters.price_projection(
            price_data=filters.price_data(ex_best_offers=True)
        )
    )
    
    if not book:
        print("No book returned.")
        return
        
    print("\n[Runner Data]")
    for runner in book[0].runners:
        print(f"ID: {runner.selection_id} | Status: {runner.status}")
        
        back_price = None
        if runner.ex.available_to_back:
            back_price = runner.ex.available_to_back[0].price
            
        print(f"  > Back Price: {back_price}")
        print(f"  > LTP: {runner.last_price_traded}")
        
        # Check matching
        matched_price = odds.get(runner.selection_id)
        print(f"  > RESOLVED PRICE IN MAP: {matched_price}")
        
    fetcher.logout()

if __name__ == "__main__":
    debug_live_market()
