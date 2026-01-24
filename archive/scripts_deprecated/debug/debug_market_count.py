import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration.betfair_fetcher import BetfairOddsFetcher
from datetime import datetime, timedelta
from betfairlightweight import filters

def debug_market_count():
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        return

    print("1. Default Filter (Now-2h to +24h, AU, WIN)")
    markets = fetcher.get_greyhound_markets()
    print(f"   Count: {len(markets)}")
    if markets:
        print(f"   Sample: {markets[0].market_name} Time: {markets[0].market_start_time}")

    print("\n2. WIDE Filter (Now-12h to +36h, AU, WIN)")
    # Custom filter
    mf = filters.market_filter(
        event_type_ids=['4339'],
        market_countries=['AU'],
        market_type_codes=['WIN'],
        market_start_time={
            'from': (datetime.now() - timedelta(hours=12)).isoformat(),
            'to': (datetime.now() + timedelta(hours=36)).isoformat()
        }
    )
    markets = fetcher.trading.betting.list_market_catalogue(
        filter=mf,
        max_results=200,
        sort='FIRST_TO_START'
    )
    print(f"   Count: {len(markets)}")

    print("\n3. NO COUNTRY Filter (Now-2h to +24h, WIN)")
    mf = filters.market_filter(
        event_type_ids=['4339'],
        market_type_codes=['WIN'],
        market_start_time={
            'from': (datetime.now() - timedelta(hours=2)).isoformat(),
            'to': (datetime.now() + timedelta(hours=24)).isoformat()
        }
    )
    markets = fetcher.trading.betting.list_market_catalogue(
        filter=mf,
        max_results=200,
        sort='FIRST_TO_START'
    )
    print(f"   Count: {len(markets)}")
    
    fetcher.logout()

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    debug_market_count()
