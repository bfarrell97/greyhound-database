"""
Debug script to see what greyhound markets Betfair actually has
"""
from betfair_odds_fetcher import BetfairOddsFetcher
from datetime import datetime

print("="*80)
print("BETFAIR MARKETS DEBUG")
print("="*80)

fetcher = BetfairOddsFetcher()

# Login
print("\nLogging in to Betfair...")
if not fetcher.login():
    print("[ERROR] Failed to login to Betfair")
    exit(1)

print("[OK] Logged in successfully")

# Get greyhound markets specifically for 08/12/2025 (today)
print("\nFetching greyhound markets for 08/12/2025...")
target_date = datetime(2025, 12, 8)
from_time = target_date.replace(hour=0, minute=0, second=0)
to_time = target_date.replace(hour=23, minute=59, second=59)
print(f"Time range: {from_time} to {to_time}")

markets = fetcher.get_greyhound_markets(from_time=from_time, to_time=to_time)
print(f"[OK] Found {len(markets)} greyhound markets for 08/12/2025")

# Display all markets with event names
print("\n" + "="*80)
print("ALL GREYHOUND MARKETS")
print("="*80)

for i, market in enumerate(markets, 1):
    event_name = market.event.name if hasattr(market, 'event') else "Unknown"
    market_name = market.market_name
    market_time = market.market_start_time if hasattr(market, 'market_start_time') else "Unknown"

    print(f"\n{i}. Event: {event_name}")
    print(f"   Market: {market_name}")
    print(f"   Time: {market_time}")
    print(f"   Market ID: {market.market_id}")

# Check specifically for Sandown
print("\n" + "="*80)
print("SANDOWN MARKETS")
print("="*80)

sandown_markets = [m for m in markets if 'sandown' in (m.event.name if hasattr(m, 'event') else "").lower()]
print(f"\nFound {len(sandown_markets)} Sandown markets")

for market in sandown_markets:
    event_name = market.event.name if hasattr(market, 'event') else "Unknown"
    market_name = market.market_name
    print(f"\n  Event: {event_name}")
    print(f"  Market: {market_name}")

# Test the normalization
print("\n" + "="*80)
print("TRACK NAME NORMALIZATION TEST")
print("="*80)

import re
test_names = [
    "Sandown (SAP)",
    "The Meadows (MEP)",
    "Murray Bridge (MBR)",
    "Angle Park",
    "Healesville"
]

for name in test_names:
    normalized = re.sub(r'\s*\([A-Z]{3}\)\s*$', '', name).strip()
    print(f"  '{name}' -> '{normalized}'")

print("\n" + "="*80)
