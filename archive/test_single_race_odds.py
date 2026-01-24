"""Test odds fetching for a single race"""
from betfair_api import BetfairAPI
from config import BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD

api = BetfairAPI(BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD)

print("Logging in...")
api.login()

print("\nFetching odds for Nowra R1...")
odds = api.get_race_odds("Nowra", 1)

print(f"\nFinal odds map:")
for box, price in sorted(odds.items()):
    print(f"  Box {box}: ${price:.2f}")
