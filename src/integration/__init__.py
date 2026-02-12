"""External API integrations for Betfair Exchange and form data providers.

This package contains clients for:
- Betfair Exchange API (betting, odds, market data)
- Topaz form data API (historical race results)
- Bet scheduling (queue bets for future execution)

Modules:
    betfair_api: REST API client for Betfair Exchange
    betfair_fetcher: Odds fetching and market monitoring
    topaz_api: Historical form data provider
    bet_scheduler: Schedule bets for execution at specific times

Example:
    >>> from src.integration.betfair_api import BetfairAPI
    >>> api = BetfairAPI()
    >>> api.login()
    >>> markets = api.list_markets()
"""

from src.integration.betfair_api import BetfairAPI
from src.integration.topaz_api import TopazAPI
from src.integration.bet_scheduler import BetScheduler

__all__ = [
    'BetfairAPI',
    'TopazAPI',
    'BetScheduler',
]
