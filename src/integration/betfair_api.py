"""
Betfair API Client for Greyhound Racing Odds
Provides access to current betting odds via Betfair Exchange API
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class BetfairAPI:
    """
    Client for interacting with the Betfair Exchange API

    Documentation: https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni

    You need:
    1. Betfair account with API access enabled
    2. Application key (app key) from Betfair Developer Program
    3. Session token (obtained via login)

    To get started:
    1. Create a Betfair account at https://www.betfair.com.au/
    2. Apply for API access at https://developer.betfair.com/
    3. Get your app key from your account
    4. Add credentials to config.py
    """

    def __init__(self, app_key: str, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Betfair API client

        Args:
            app_key: Your Betfair application key
            username: Betfair account username (optional)
            password: Betfair account password (optional)
        """
        self.app_key = app_key
        self.username = username
        self.password = password
        self.session_token = None
        self.base_url = "https://api.betfair.com/exchange/betting/json-rpc/v1"
        # Australia & New Zealand endpoint as per documentation
        self.identity_url = "https://identitysso.betfair.com.au/api"

    def login(self):
        """
        Login to Betfair and get session token

        Returns:
            Session token string
        """
        if not self.username or not self.password:
            raise ValueError("Username and password required for login")

        headers = {
            'Accept': 'application/json',
            'X-Application': self.app_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        # Use /login endpoint as per official documentation
        payload = f"username={self.username}&password={self.password}"

        response = requests.post(
            f"{self.identity_url}/login",
            headers=headers,
            data=payload
        )

        print(f"[DEBUG] Login response status: {response.status_code}")

        if response.status_code != 200:
            print(f"[DEBUG] Response: {response.text[:500]}")
            raise Exception(f"Login failed (status {response.status_code})")

        try:
            result = response.json()
            print(f"[DEBUG] Login response: {result}")
        except Exception as e:
            print(f"[DEBUG] Response text: {response.text[:500]}")
            raise Exception(f"Failed to parse login response as JSON")

        # Check response format from documentation
        if result.get('status') == 'SUCCESS':
            self.session_token = result['token']
            print(f"[OK] Logged in to Betfair successfully")
            print(f"[OK] Session token: {self.session_token[:20]}...")
            return self.session_token
        else:
            error = result.get('error', 'Unknown error')
            status = result.get('status', 'FAIL')
            raise Exception(f"Login failed - Status: {status}, Error: {error}")

    def _make_request(self, method: str, params: Dict) -> Dict[str, Any]:
        """
        Make a JSON-RPC request to Betfair API

        Args:
            method: API method name (e.g., 'listMarketCatalogue')
            params: Method parameters

        Returns:
            Response dictionary
        """
        if not self.session_token:
            self.login()

        headers = {
            'X-Application': self.app_key,
            'X-Authentication': self.session_token,
            'Content-Type': 'application/json'
        }

        payload = {
            "jsonrpc": "2.0",
            "method": f"SportsAPING/v1.0/{method}",
            "params": params,
            "id": 1
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        result = response.json()

        if 'error' in result:
            raise Exception(f"API error: {result['error']}")

        return result.get('result')

    def list_events(self, event_type_id: str = "4339") -> List[Dict[str, Any]]:
        """
        List all greyhound racing events

        Args:
            event_type_id: Event type ID (4339 = Greyhound Racing)

        Returns:
            List of event dictionaries
        """
        params = {
            "filter": {
                "eventTypeIds": [event_type_id]
            }
        }

        return self._make_request("listEvents", params)

    def list_market_catalogue(self, event_id: Optional[str] = None,
                             event_type_id: str = "4339",
                             market_countries: List[str] = ["AU"],
                             market_type_codes: List[str] = ["WIN", "PLACE"]) -> List[Dict[str, Any]]:
        """
        List market catalogues for greyhound racing

        Args:
            event_id: Specific event ID (optional)
            event_type_id: Event type ID (4339 = Greyhound Racing)
            market_countries: List of country codes (default: ['AU'])
            market_type_codes: List of market types (default: ['WIN'] for win market)

        Returns:
            List of market catalogue dictionaries
        """
        filter_dict = {
            "eventTypeIds": [event_type_id],
            "marketCountries": market_countries,
            "marketTypeCodes": market_type_codes
        }

        if event_id:
            filter_dict["eventIds"] = [event_id]

        params = {
            "filter": filter_dict,
            "maxResults": "200",
            "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"]
        }

        return self._make_request("listMarketCatalogue", params)

    def get_market_prices(self, market_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get current prices for specific markets

        Args:
            market_ids: List of market IDs

        Returns:
            List of market book dictionaries with current prices
        """
        params = {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS"]
            }
        }

        return self._make_request("listMarketBook", params)

    def find_race_market(self, track_name: str, race_number: int, race_time: str) -> Optional[Dict[str, Any]]:
        """
        Find the Betfair market for a specific race

        Args:
            track_name: Track name (e.g., "The Meadows", "Sandown")
            race_number: Race number
            race_time: Race start time (for matching)

        Returns:
            Market dictionary or None if not found
        """
        # Get all greyhound markets
        markets = self.list_market_catalogue()

        # Try to match by track name and race number
        # Note: This is a heuristic - Betfair naming can vary
        for market in markets:
            market_name = market.get('marketName', '')
            event_name = market.get('event', {}).get('name', '')

            # Check if track name appears in event or market name
            if track_name.lower() in event_name.lower():
                # Check if this is the right race (usually in market name like "R1", "R2", etc.)
                if f"R{race_number}" in market_name or f"Race {race_number}" in market_name:
                    return market

        return None

    def get_race_odds(self, track_name: str, race_number: int, race_time: str = None) -> Dict[int, float]:
        """
        Get current odds for all runners in a specific race

        Args:
            track_name: Track name
            race_number: Race number
            race_time: Race start time (optional)

        Returns:
            Dictionary mapping runner position (trap/box) to current odds
        """
        # Find the market
        market = self.find_race_market(track_name, race_number, race_time)

        if not market:
            print(f"    [WARNING] Market not found for {track_name} R{race_number}")
            return {}

        market_id = market['marketId']
        print(f"    [DEBUG] Found market ID {market_id} for {track_name} R{race_number}")

        # Get current prices
        market_books = self.get_market_prices([market_id])

        if not market_books:
            print(f"    [WARNING] No price data returned for {market_id}")
            return {}

        market_book = market_books[0]

        # Extract odds for each runner
        odds_map = {}

        for runner in market_book.get('runners', []):
            # Get selection ID
            selection_id = runner.get('selectionId')

            # Find corresponding runner in market catalogue
            for catalog_runner in market.get('runners', []):
                if catalog_runner.get('selectionId') == selection_id:
                    # Extract trap/box number from metadata
                    metadata = catalog_runner.get('metadata', {})

                    # Try different metadata keys for box/trap number
                    trap = None
                    for key in ['TRAP', 'CLOTH_NUMBER', 'CLOTH', 'STALL_DRAW']:
                        if key in metadata:
                            try:
                                trap = int(metadata[key])
                                break
                            except (ValueError, TypeError):
                                continue

                    # Fallback: try to parse from runner name (e.g., "1. Dog Name")
                    if trap is None and 'runnerName' in catalog_runner:
                        runner_name = catalog_runner['runnerName']
                        if '.' in runner_name:
                            try:
                                trap = int(runner_name.split('.')[0].strip())
                            except (ValueError, TypeError):
                                pass

                    if trap:
                        # Get best available back odds
                        ex = runner.get('ex', {})
                        available_to_back = ex.get('availableToBack', [])

                        # DEBUG: Print first runner to see structure
                        if len(odds_map) == 0:
                            print(f"    [DEBUG] Sample runner structure:")
                            print(f"      selectionId: {selection_id}")
                            print(f"      trap: {trap}")
                            print(f"      ex: {ex}")
                            print(f"      availableToBack: {available_to_back}")

                        if available_to_back:
                            # Get best back price (first in list)
                            best_price = available_to_back[0].get('price')
                            if best_price:
                                odds_map[int(trap)] = float(best_price)
                                print(f"      Trap {trap}: ${best_price}")

                    break

        print(f"    [DEBUG] Extracted odds for {len(odds_map)} runners")
        return odds_map


def test_connection():
    """
    Test Betfair API connection

    NOTE: You need to add your credentials to config.py first:

    BETFAIR_APP_KEY = "your_app_key_here"
    BETFAIR_USERNAME = "your_username_here"
    BETFAIR_PASSWORD = "your_password_here"
    """
    try:
        from config import BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD
    except ImportError:
        print("[ERROR] Betfair credentials not found in config.py")
        print("\nPlease add the following to config.py:")
        print("  BETFAIR_APP_KEY = 'your_app_key_here'")
        print("  BETFAIR_USERNAME = 'your_username_here'")
        print("  BETFAIR_PASSWORD = 'your_password_here'")
        print("\nSee https://developer.betfair.com/ for how to get credentials")
        return

    api = BetfairAPI(BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD)

    print("Testing Betfair API connection...")

    # Login
    api.login()

    # Get greyhound events
    events = api.list_events()
    print(f"\n[OK] Found {len(events)} greyhound racing events")

    # Get markets
    markets = api.list_market_catalogue()
    print(f"[OK] Found {len(markets)} markets")

    if markets:
        print("\nSample market:")
        sample = markets[0]
        print(f"  Event: {sample.get('event', {}).get('name')}")
        print(f"  Market: {sample.get('marketName')}")
        print(f"  Start time: {sample.get('marketStartTime')}")
        print(f"  Runners: {len(sample.get('runners', []))}")


if __name__ == "__main__":
    test_connection()
