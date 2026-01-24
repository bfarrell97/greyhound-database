"""Quick check of Betfair odds availability"""
from betfair_odds_fetcher import BetfairOddsFetcher
from datetime import datetime, timedelta
from betfairlightweight import filters

fetcher = BetfairOddsFetcher()
print('Logging in to Betfair...')
if not fetcher.login():
    print('Login failed')
    exit()

# Get today's markets
today = datetime.now()
from_time = today.replace(hour=0, minute=0, second=0)
to_time = today.replace(hour=23, minute=59, second=59)

print(f'Fetching markets for {today.strftime("%Y-%m-%d")}...')
markets = fetcher.get_greyhound_markets(from_time, to_time)
print(f'Found {len(markets)} markets')

if markets:
    # Get odds for first few markets
    market_ids = [m.market_id for m in markets[:10]]
    
    market_books = fetcher.trading.betting.list_market_book(
        market_ids=market_ids,
        price_projection=filters.price_projection(
            price_data=filters.price_data(ex_best_offers=True)
        )
    )
    
    print(f'\nSample odds from first 10 markets:')
    for book in market_books:
        market = next((m for m in markets if m.market_id == book.market_id), None)
        if market:
            print(f'\n{market.event.name} - {market.market_name} (Start: {market.market_start_time})')
        
        odds_list = []
        for runner in book.runners:
            if runner.ex and runner.ex.available_to_back:
                odds = runner.ex.available_to_back[0].price
                odds_list.append(odds)
        
        if odds_list:
            print(f'  Odds range: ${min(odds_list):.2f} - ${max(odds_list):.2f}')
            print(f'  In $1.50-$30 range: {sum(1 for o in odds_list if 1.5 <= o <= 30)}/{len(odds_list)}')
        else:
            print(f'  No odds available yet')

fetcher.logout()
print('\nDone')
