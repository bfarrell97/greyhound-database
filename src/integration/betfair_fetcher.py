"""
Betfair Odds Fetcher using betfairlightweight library
Simpler and more reliable than custom API implementation
"""

import betfairlightweight
from betfairlightweight import filters
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.core.config import BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD
import threading
import time


class BetfairOddsFetcher:
    """Fetch current odds from Betfair using betfairlightweight"""

    def __init__(self):
        """Initialize Betfair API client"""
        self.trading = betfairlightweight.APIClient(
            username=BETFAIR_USERNAME,
            password=BETFAIR_PASSWORD,
            app_key=BETFAIR_APP_KEY,
            certs="",  # Empty string to avoid looking for /certs directory
            cert_files=None,  # Don't use SSL certificates, use username/password
        )
        # Keepalive control
        self._keepalive_thread = None
        self._keepalive_running = False
        self._keepalive_interval = 900  # seconds (default 15 minutes)
        self._login_lock = threading.Lock()

    def login(self):
        """Login to Betfair using interactive (username/password) method"""
        with self._login_lock:
            try:
                # Use login_interactive instead of login (which requires certificates)
                self.trading.login_interactive()
                print("[OK] Logged in to Betfair successfully")
                # Start keepalive thread to maintain session
                try:
                    self.start_keepalive()
                except Exception:
                    pass
                return True
            except Exception as e:
                print(f"[ERROR] Betfair login failed: {e}")
                return False

    def logout(self):
        """Logout from Betfair"""
        try:
            # Stop keepalive to avoid races during logout
            self.stop_keepalive()
            self.trading.logout()
            print("[OK] Logged out from Betfair")
        except:
            pass

    def keep_alive(self):
        """Send keep alive to Betfair to maintain session"""
        try:
            self.trading.keep_alive()
            # print("[DEBUG] Betfair Keep Alive sent.")
            return True
        except Exception as e:
            print(f"[WARN] Keep Alive failed: {e}")
            return False

    def start_keepalive(self, interval_seconds: int = None):
        """Start a background thread that periodically calls keep_alive()."""
        if interval_seconds is None:
            interval_seconds = self._keepalive_interval
        # If thread already running, update interval and return
        if self._keepalive_running:
            self._keepalive_interval = interval_seconds
            return

        self._keepalive_running = True
        self._keepalive_interval = interval_seconds

        def _loop():
            while self._keepalive_running:
                time.sleep(self._keepalive_interval)
                try:
                    ok = self.keep_alive()
                    if not ok:
                        print("[WARN] Keep Alive returned False; attempting re-login")
                        try:
                            # Try to re-login once
                            self.login()
                        except Exception as e:
                            print(f"[ERROR] Re-login attempt failed: {e}")
                except Exception as e:
                    print(f"[ERROR] Exception in keepalive loop: {e}")

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self):
        """Stop the keepalive thread if running."""
        self._keepalive_running = False
        if self._keepalive_thread:
            try:
                self._keepalive_thread.join(timeout=2)
            except Exception:
                pass
            self._keepalive_thread = None

    def _safe_call(self, call_fn, fallback=None):
        """Wrapper for Betfair API calls that retries on INVALID_SESSION_INFORMATION errors.

        Args:
            call_fn: Zero-arg callable performing the API call.
            fallback: Value to return if call cannot succeed after retry.
        """
        try:
            return call_fn()
        except Exception as e:
            # Detect APIError / invalid session
            msg = ''
            try:
                # Prefer structured response when available
                if isinstance(e, betfairlightweight.exceptions.APIError):
                    resp = getattr(e, 'response', None)
                    if isinstance(resp, str):
                        msg = resp
                    elif isinstance(resp, dict):
                        msg = str(resp.get('error', ''))
                    elif e.args:
                        # Fallback to args (some APIError constructions use a message string)
                        msg = ' '.join(str(a) for a in e.args if isinstance(a, (str, int, float)))
                else:
                    msg = str(e)
            except Exception:
                msg = ''

            if 'INVALID_SESSION_INFORMATION' in msg or 'ANGX-0003' in msg or 'INVALID_SESSION' in msg:
                print("[WARN] Betfair session invalid, attempting re-login...")
                try:
                    if self.login():
                        try:
                            return call_fn()
                        except Exception as e2:
                            print(f"[ERROR] API call failed after re-login: {e2}")
                            return fallback
                    else:
                        print("[ERROR] Re-login failed; will return fallback")
                        return fallback
                except Exception as e:
                    print(f"[ERROR] Exception while re-logging in: {e}")
                    return fallback
            # Not a session error - re-raise for upper-level handlers to inspect
            raise

    def get_nearest_tick(self, price: float) -> float:
        """
        Snap a price to the nearest valid Betfair tick size.
        Rules:
            1.01 - 2.0: 0.01
            2.0 - 3.0: 0.02
            3.0 - 4.0: 0.05
            4.0 - 6.0: 0.1
            6.0 - 10.0: 0.2
            10.0 - 20.0: 0.5
            20.0 - 30.0: 1.0
            30.0 - 50.0: 2.0
            50.0 - 100.0: 5.0
            100.0 - 1000.0: 10.0
        """
        if not price or price < 1.01: return 1.01
        if price >= 1000.0: return 1000.0
        
        if price < 2.0: tick = 0.01
        elif price < 3.0: tick = 0.02
        elif price < 4.0: tick = 0.05
        elif price < 6.0: tick = 0.1
        elif price < 10.0: tick = 0.2
        elif price < 20.0: tick = 0.5
        elif price < 30.0: tick = 1.0
        elif price < 50.0: tick = 2.0
        elif price < 100.0: tick = 5.0
        else: tick = 10.0
        
        # Robust rounding to nearest tick
        return round(round(price / tick) * tick, 2)

    def get_tick_size(self, price: float) -> float:
        """Get the valid Betfair tick size for a given price."""
        if price < 2.0: return 0.01
        if price < 3.0: return 0.02
        if price < 4.0: return 0.05
        if price < 6.0: return 0.1
        if price < 10.0: return 0.2
        if price < 20.0: return 0.5
        if price < 30.0: return 1.0
        if price < 50.0: return 2.0
        if price < 100.0: return 5.0
        return 10.0

    def get_next_tick(self, price: float, num_ticks: int = 1) -> float:
        """
        Move a price by a certain number of ticks (positive for up, negative for down).
        """
        current_price = self.get_nearest_tick(price)
        for _ in range(abs(num_ticks)):
            tick = self.get_tick_size(current_price if num_ticks > 0 else current_price - 0.001)
            if num_ticks > 0:
                current_price += tick
            else:
                current_price -= tick
            
            if current_price < 1.01: return 1.01
            if current_price > 1000.0: return 1000.0
            
            # Snap again to be sure
            current_price = self.get_nearest_tick(current_price)
            
        return current_price

    def get_greyhound_markets(self, from_time: Optional[datetime] = None,
                             to_time: Optional[datetime] = None,
                             market_type_codes: List[str] = ['WIN', 'PLACE']) -> List:
        """
        Get greyhound racing markets

        Args:
            from_time: Start time for markets (default: now)
            to_time: End time for markets (default: 24 hours from now)

        Returns:
            List of market catalogues
        """
        if from_time is None:
            # Look back 2 hours to catch delayed races or just-started ones
            # Use UTC as Betfair API expects UTC
            from_time = datetime.utcnow() - timedelta(hours=2)

        if to_time is None:
            to_time = datetime.utcnow() + timedelta(hours=24)

        # Create filter for greyhound racing markets
        # print(f"[DEBUG] Querying markets from {from_time.isoformat()} to {to_time.isoformat()}")
        market_filter = filters.market_filter(
            event_type_ids=['4339'],  # Greyhound Racing
            market_countries=['AU'],  # Australia
            market_type_codes=market_type_codes,
            market_start_time={
                'from': from_time.isoformat(),
                'to': to_time.isoformat()
            }
        )

        try:
            # STEP 1: Lightweight fetch to get ALL Market IDs
            # We ONLY ask for IDs and Start Times to keep response small
            initial_catalogues = self._safe_call(lambda: self.trading.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=['MARKET_START_TIME'],
                max_results=200, # Max allowed by API
                sort='FIRST_TO_START'
            ), fallback=[])
            
            if not initial_catalogues:
                print(f"[DEBUG] Betfair: No markets found ({from_time.strftime('%H:%M')} -> {to_time.strftime('%H:%M')})")
                return []
                
            all_market_ids = [m.market_id for m in initial_catalogues]
            all_full_catalogues = []
            
            # STEP 2: Batch fetch Full Details (to avoid TOO_MUCH_DATA error)
            # Batch size of 40 is safe for full description data
            BATCH_SIZE = 40
            
            for i in range(0, len(all_market_ids), BATCH_SIZE):
                batch_ids = all_market_ids[i:i + BATCH_SIZE]
                
                batch_filter = filters.market_filter(
                    market_ids=batch_ids
                )
                
                batch_cats = self._safe_call(lambda: self.trading.betting.list_market_catalogue(
                    filter=batch_filter,
                    market_projection=[
                        'RUNNER_DESCRIPTION',
                        'EVENT',
                        'MARKET_START_TIME',
                        'RUNNER_METADATA',
                        'MARKET_DESCRIPTION'  # Heavy field, hence batching
                    ],
                    max_results=BATCH_SIZE
                ), fallback=[])

                if batch_cats:
                    all_full_catalogues.extend(batch_cats)
            
            market_catalogues = all_full_catalogues
            
            # Formatter for cleaner logs
            t_fmt = "%H:%M"
            from_str = from_time.strftime(t_fmt)
            to_str = to_time.strftime(t_fmt)
            
            count = len(market_catalogues)
            if count > 0:
                print(f"[DEBUG] Betfair: Found {count} markets ({from_str} -> {to_str})")
            
            return market_catalogues
        except Exception:
            print("[ERROR] Failed to get markets: <exception caught> - see traceback")
            import traceback
            traceback.print_exc()
            return []

    def get_market_prices(self, market_id: str) -> Dict[int, Dict[str, float]]:
        """
        Get current back and lay prices for a specific market
        
        Returns:
            Dictionary mapping runner selection ID to {'back': price, 'lay': price}
        """
        try:
            market_book = self.trading.betting.list_market_book(
                market_ids=[market_id],
                price_projection=filters.price_projection(
                    price_data=filters.price_data(ex_best_offers=True)
                )
            )

            if not market_book:
                return {}

            prices_map = {}
            for runner in market_book[0].runners:
                back_price = None
                lay_price = None
                
                # 1. Best Back Price
                if runner.ex and runner.ex.available_to_back:
                    back_price = float(runner.ex.available_to_back[0].price)
                elif runner.last_price_traded:
                    back_price = float(runner.last_price_traded)
                
                # 2. Best Lay Price
                if runner.ex and runner.ex.available_to_lay:
                    lay_price = float(runner.ex.available_to_lay[0].price)

                # 3. LTP
                ltp = float(runner.last_price_traded) if runner.last_price_traded else None
                
                prices_map[runner.selection_id] = {
                    'back': back_price,
                    'lay': lay_price,
                    'ltp': ltp
                }

            return prices_map
        except Exception as e:
            print(f"[ERROR] Failed to get market prices: {e}")
            return {}

    def get_all_market_prices(self, market_ids: List[str]) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Get back and lay prices for multiple markets in a single API call (FAST)
        
        Returns:
            Dictionary mapping market_id to {selection_id: {'back': price, 'lay': price}}
        """
        if not market_ids:
            return {}
            
        try:
            # Betfair allows up to 40 markets per request
            market_books = self.trading.betting.list_market_book(
                market_ids=market_ids[:40],
                price_projection=filters.price_projection(
                    price_data=filters.price_data(ex_best_offers=True),
                    virtualise=True # Match Website Cross-Matching prices
                )
            )

            all_prices = {}
            for book in market_books:
                prices_map = {}
                for runner in book.runners:
                    back_price = None
                    lay_price = None
                    
                    if runner.ex and runner.ex.available_to_back:
                        back_price = float(runner.ex.available_to_back[0].price)
                    # REMOVED LTP FALLBACK: If no liquidity, return None (don't use stale LTP)
                    
                    if runner.ex and runner.ex.available_to_lay:
                        lay_price = float(runner.ex.available_to_lay[0].price)
                    
                    prices_map[runner.selection_id] = {
                        'back': back_price,
                        'lay': lay_price
                    }
                all_prices[book.market_id] = prices_map
            
            return all_prices
        except Exception as e:
            print(f"[ERROR] Batch market prices failed: {e}")
            return {}

    def get_market_odds(self, market_id: str, price_type: str = 'BACK') -> Dict[int, float]:
        """
        Get current odds for a specific market
        
        Args:
            market_id: Betfair market ID
            price_type: 'BACK' (Best Available) or 'LTP' (Last Traded Price)

        Returns:
            Dictionary mapping runner selection ID to price
        """
        try:
            # Get market prices
            market_book = self.trading.betting.list_market_book(
                market_ids=[market_id],
                price_projection=filters.price_projection(
                    price_data=filters.price_data(ex_best_offers=True)
                )
            )

            if not market_book:
                return {}

            odds_map = {}
            for runner in market_book[0].runners:
                price = None
                
                # LTP REQUEST
                if price_type == 'LTP':
                    if runner.last_price_traded:
                        price = float(runner.last_price_traded)
                    # Fallback to Back price if LTP missing? Or None?
                    # Let's fallback to Back if LTP is missing to ensure we get SOMETHING
                    elif runner.ex and runner.ex.available_to_back:
                        price = float(runner.ex.available_to_back[0].price)
                
                # BACK PRICE REQUEST (Default)
                else:
                    if runner.ex and runner.ex.available_to_back:
                        price = float(runner.ex.available_to_back[0].price)
                    elif runner.last_price_traded:
                        price = float(runner.last_price_traded)

                if price:
                    odds_map[runner.selection_id] = price

            return odds_map

        except Exception as e:
            print(f"[ERROR] Failed to get odds for market {market_id}: {e}")
            return {}

    def find_race_market(self, track_name: str, race_number: int,
                        race_time: Optional[str] = None) -> Optional[tuple]:
        """
        Find Betfair market for a specific race

        Args:
            track_name: Track name (e.g., "Geelong", "The Meadows")
            race_number: Race number
            race_time: Race start time (optional, for better matching)

        Returns:
            Tuple of (market_id, selection_id_to_trap_map) or None
        """
        # Get markets for today and tomorrow (full days)
        # This ensures we get all races for today even if they start late in the evening
        now = datetime.now()
        from_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        to_time = (now + timedelta(days=2)).replace(hour=23, minute=59, second=59, microsecond=0)
        markets = self.get_greyhound_markets(from_time=from_time, to_time=to_time)

        # Normalize track name: remove track codes in parentheses like " (SAP)", " (MEP)", etc.
        import re
        normalized_track = re.sub(r'\s*\([A-Z]{3}\)\s*$', '', track_name).strip()

        # Special mapping: Betfair uses different names for some tracks
        track_mapping = {
            'Sandown': 'Sandown Park',
        }

        # Apply special mapping if exists
        search_track = track_mapping.get(normalized_track, normalized_track)

        for market in markets:
            event_name = market.event.name if hasattr(market, 'event') else ""
            market_name = market.market_name

            # Check if track name matches
            if search_track.lower() in event_name.lower():
                # Check if race number matches (usually in market name like "R1", "R2", etc.)
                if f"R{race_number}" in market_name or f"Race {race_number}" in market_name:
                    # Build selection ID to trap/box mapping
                    selection_to_trap = {}
                    for runner in market.runners:
                        # Extract trap number from metadata or runner name
                        trap = None
                        if hasattr(runner, 'metadata') and runner.metadata:
                            trap = runner.metadata.get('TRAP')

                        # If no TRAP in metadata, parse from runner name (e.g., "1. Corinthian" -> trap 1)
                        if not trap and hasattr(runner, 'runner_name'):
                            import re
                            match = re.match(r'^(\d+)\.\s', runner.runner_name)
                            if match:
                                trap = match.group(1)

                        if trap:
                            selection_to_trap[runner.selection_id] = int(trap)

                    return (market.market_id, selection_to_trap)

        return None

    def get_market_names(self, market_ids: List[str]) -> Dict[str, str]:
        """Get market names for a list of market IDs"""
        if not market_ids:
            return {}
            
        try:
            market_filter = filters.market_filter(market_ids=market_ids)
            market_catalogues = self.trading.betting.list_market_catalogue(
                filter=market_filter,
                max_results=len(market_ids)
            )
            
            return {m.market_id: m.market_name for m in market_catalogues}
        except Exception as e:
            print(f"[ERROR] Failed to get market names: {e}")
            return {}

    def get_current_orders(self, market_id: str = None) -> List:
        """Get current unmatched/matched orders"""
        try:
            current_orders = self.trading.betting.list_current_orders(
                market_ids=[market_id] if market_id else None
            )
            if hasattr(current_orders, 'orders'):
                return current_orders.orders
            elif hasattr(current_orders, 'current_orders'):
                return current_orders.current_orders
            else:
                 # Fallback: inspect what we got
                 print(f"[WARN] potential structure mismatch in list_current_orders: {dir(current_orders)}")
                 return []
        except Exception as e:
            print(f"[ERROR] Failed to get current orders: {e}")
            return []

    def cancel_bet(self, market_id: str, bet_id: str = None) -> Dict:
        """
        Cancel an unmatched bet or unmatched portion of a partially matched bet.
        
        Args:
            market_id: Betfair market ID
            bet_id: Optional specific bet ID to cancel. If None, cancels ALL unmatched on market.
            
        Returns:
            Dict with 'is_success', 'size_cancelled', 'error' keys
        """
        try:
            if bet_id:
                # Cancel specific bet
                cancel_instruction = {
                    'betId': bet_id
                }
                result = self.trading.betting.cancel_orders(
                    market_id=market_id,
                    instructions=[cancel_instruction]
                )
            else:
                # Cancel ALL unmatched on market
                result = self.trading.betting.cancel_orders(
                    market_id=market_id
                )
            
            # Parse result
            if result and hasattr(result, 'instruction_reports'):
                reports = result.instruction_reports
                if reports:
                    report = reports[0]
                    if report.status == 'SUCCESS':
                        return {
                            'is_success': True,
                            'size_cancelled': float(report.size_cancelled) if hasattr(report, 'size_cancelled') else 0,
                            'error': None
                        }
                    else:
                        return {
                            'is_success': False,
                            'size_cancelled': 0,
                            'error': getattr(report, 'error_code', 'UNKNOWN')
                        }
            
            return {'is_success': False, 'size_cancelled': 0, 'error': 'No result'}
            
        except Exception as e:
            print(f"[ERROR] Cancel bet failed: {e}")
            return {'is_success': False, 'size_cancelled': 0, 'error': str(e)}

    def get_market_result(self, market_id: str) -> Dict:
        """
        Get the result of a closed market (Winners and BSP).
        Returns a dict mapping selection_id -> {'status': 'WINNER'/'LOSER', 'bsp': float}
        """
        try:
            market_book = self.trading.betting.list_market_book(
                market_ids=[market_id],
                price_projection=filters.price_projection(
                    price_data=filters.price_data(sp_available=True), # Request SP
                    virtualise=True
                )
            )

            if not market_book:
                return {}

            results = {}
            for runner in market_book[0].runners:
                # BSP
                bsp = None
                if hasattr(runner, 'sp') and runner.sp:
                     # actual_sp is the finalized BSP
                     bsp = float(runner.sp.actual_sp) if runner.sp.actual_sp else None
                
                results[runner.selection_id] = {
                    'status': runner.status, # WINNER, LOSER, REMOVED
                    'bsp': bsp
                }
            return results

        except Exception as e:
            print(f"[ERROR] Failed to get market result: {e}")
            return {}

    def get_cleared_orders(self, bet_status='SETTLED', days=3) -> List:
        """Get settled/voided orders history"""
        if not self.trading: return []
        
        try:
            from betfairlightweight import filters
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            # If bet_status is provided, filter by it. If None, fetch ALL cleared orders (Settled, Voided, Lapsed, Cancelled).
            kwargs = {}
            if bet_status:
                kwargs['bet_status'] = bet_status

            cleared_orders = self.trading.betting.list_cleared_orders(
                settled_date_range=filters.time_range(from_=from_date),
                **kwargs
            )
            
            if hasattr(cleared_orders, 'orders'):
                return cleared_orders.orders
            elif hasattr(cleared_orders, 'cleared_orders'):
                return cleared_orders.cleared_orders
            else:
                 return []
                 
        except Exception as e:
            print(f"[ERROR] Failed to get cleared orders: {e}")
            return []
            
    def get_account_funds(self) -> Dict:
        """Get account balance"""
        try:
            funds = self.trading.account.get_account_funds()
            return {
                'available': funds.available_to_bet_balance,
                'exposure': funds.exposure,
                'total': funds.wallet,
                'points': funds.points_balance
            }
        except Exception as e:
            print(f"[ERROR] Failed to get funds: {e}")
            return {'available': 0.0, 'exposure': 0.0, 'total': 0.0, 'points': 0}

    def get_market_details(self, market_id: str, selection_id: int = None) -> dict:
        """
        Get market name, runner name, and race time for a given market/selection ID.
        
        Returns:
            dict with 'market_name', 'runner_name', 'race_time'
        """
        try:
            cats = self.trading.betting.list_market_catalogue(
                filter={'marketIds': [market_id]},
                market_projection=['EVENT', 'RUNNER_DESCRIPTION', 'MARKET_START_TIME'],
                max_results=1
            )
            
            if not cats:
                return {'market_name': None, 'runner_name': None, 'race_time': None}
                
            cat = cats[0]
            
            # Market name: Event Name (e.g., "Bendigo R9")
            event_name = cat.event.name if hasattr(cat, 'event') and cat.event else ''
            market_name = event_name if event_name else cat.market_name
            
            # Race time from market start time
            race_time = None
            if hasattr(cat, 'market_start_time') and cat.market_start_time:
                # Convert to local time and extract HH:MM
                from datetime import timezone
                utc_time = cat.market_start_time
                if utc_time.tzinfo is None:
                    utc_time = utc_time.replace(tzinfo=timezone.utc)
                local_time = utc_time.astimezone()
                race_time = local_time.strftime('%H:%M')
            
            runner_name = None
            if selection_id and hasattr(cat, 'runners'):
                for runner in cat.runners:
                    if runner.selection_id == selection_id:
                        runner_name = runner.runner_name
                        break
                        
            return {
                'market_name': market_name,
                'runner_name': runner_name,
                'race_time': race_time
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get market details: {e}")
            return {'market_name': None, 'runner_name': None, 'race_time': None}

    def get_race_odds_by_box(self, track_name: str, race_number: int,
                            race_time: Optional[str] = None, price_type: str = 'BACK') -> Dict[int, float]:
        """
        Get current odds for a specific race, mapped by box number
        """
        result = self.find_race_market(track_name, race_number, race_time)

        if not result:
            print(f"[WARNING] Market not found for {track_name} R{race_number}")
            return {}

        market_id, selection_to_trap = result

        # Get odds for this market (keyed by selection ID)
        selection_odds = self.get_market_odds(market_id, price_type=price_type)

        # Convert to box number -> odds mapping
        box_odds = {}
        for selection_id, odds in selection_odds.items():
            if selection_id in selection_to_trap:
                box = selection_to_trap[selection_id]
                box_odds[box] = odds

        return box_odds


    def resolve_selection_id(self, track_name: str, race_number: int, dog_name: str) -> Optional[tuple]:
        """
        Resolve market_id and selection_id for a dog
        """
        # 1. Find Market
        result = self.find_race_market(track_name, race_number)
        if not result:
            print(f"Market not found for {track_name} R{race_number}")
            return None
            
        market_id, selection_to_trap = result
        
        # 2. Find Runner in Market
        try:
            # We need runner definition to get name
            cat = self.trading.betting.list_market_catalogue(
                filter=filters.market_filter(market_ids=[market_id]),
                market_projection=['RUNNER_DESCRIPTION'],
                max_results=1
            )
            
            if not cat:
                return None
                
            clean_target = dog_name.lower().replace("'", "").replace(".", "").strip()
            
            for runner in cat[0].runners:
                r_name = runner.runner_name.lower().replace("'", "").replace(".", "").strip()
                # Check for "1. Name" format
                if " " in r_name and r_name[0].isdigit():
                     parts = r_name.split(" ", 1)
                     if len(parts) > 1:
                         r_name = parts[1].strip()
                         
                if r_name == clean_target:
                    return (market_id, runner.selection_id)
                    
            # Fallback: check containment
            for runner in cat[0].runners:
                r_name = runner.runner_name.lower().replace("'", "").replace(".", "").strip()
                if clean_target in r_name or r_name in clean_target:
                     return (market_id, runner.selection_id)
                     
            print(f"Runner {dog_name} not found in {track_name} R{race_number}")
            return None
            
        except Exception as e:
            print(f"Error resolving selection: {e}")
            return None

    def place_lay_bet(self, market_id: str, selection_id: int, stake: float, price: float = None) -> Dict:
        """
        Place a LAY bet.
        If price is provided => LIMIT order (Exchange).
        If price is None => BSP Market On Close (Liability = stake).
        """
        try:
            # Fixed liability for all LAY bets (per user request)
            LAY_FIXED_LIABILITY = 10.0
            if price:
                # LIMIT ORDER: calculate size such that Liability = LAY_FIXED_LIABILITY
                size = max(round(LAY_FIXED_LIABILITY / (price - 1.0), 2), 0.01)
                instruction = filters.place_instruction(
                    order_type='LIMIT',
                    selection_id=selection_id,
                    side='LAY',
                    limit_order=filters.limit_order(
                        size=size,
                        price=price,
                        persistence_type='LAPSE'
                    )
                )
            else:
                # MARKET ON CLOSE: supply liability directly
                instruction = filters.place_instruction(
                    order_type='MARKET_ON_CLOSE',
                    selection_id=selection_id,
                    side='LAY',
                    market_on_close_order=filters.market_on_close_order(
                        liability=LAY_FIXED_LIABILITY
                    )
                )
            
            place_orders = self.trading.betting.place_orders(
                market_id=market_id,
                instructions=[instruction],
                customer_strategy_ref='AlphaEngine'
            )
            
            status = place_orders.status
            if status == 'SUCCESS':
                if hasattr(place_orders, 'place_instruction_reports'):
                    report = place_orders.place_instruction_reports[0]
                elif hasattr(place_orders, 'instruction_reports'):
                    report = place_orders.instruction_reports[0]
                else:
                    return {'status': 'FAILURE', 'error': 'Missing instruction_reports', 'is_success': False}

                return {
                    'status': report.status,
                    'bet_id': report.bet_id,
                    'avg_price_matched': report.average_price_matched,
                    'size_matched': report.size_matched,
                    'is_success': report.status == 'SUCCESS'
                }
            else:
                print(f"Place order failed: {place_orders.error_code}")
                # Try to get inner error
                if hasattr(place_orders, 'place_instruction_reports') and place_orders.place_instruction_reports:
                    rpt = place_orders.place_instruction_reports[0]
                    print(f"       Inner error: {rpt.error_code}")
                elif hasattr(place_orders, 'instruction_reports') and place_orders.instruction_reports:
                    rpt = place_orders.instruction_reports[0]
                    print(f"       Inner error: {rpt.error_code}")
                    
                return {'status': 'FAILURE', 'error': place_orders.error_code, 'is_success': False}
                
        except Exception as e:
            print(f"Error placing bet: {e}")
            return {'status': 'ERROR', 'error': str(e), 'is_success': False}

    def place_back_bet(self, market_id: str, selection_id: int, stake: float, price: float) -> Dict:
        """
        Place a BACK bet at a specified price.
        
        Args:
            market_id: Betfair market ID
            selection_id: Runner selection ID
            stake: Stake amount (e.g., 1.0 for $1)
            price: Back price (odds) - REQUIRED for back bets
            
        Returns:
            Dict with bet result including is_success, bet_id, etc.
        """
        try:
            if price:
                # LIMIT ORDER (Fixed Price)
                instruction = filters.place_instruction(
                    order_type='LIMIT',
                    selection_id=selection_id,
                    side='BACK',
                    limit_order=filters.limit_order(
                        size=stake,
                        price=price,
                        persistence_type='LAPSE'
                    )
                )
            else:
                # BSP ORDER (Market On Close)
                # For Back Bets, 'liability' in MOC order refers to the STAKE.
                instruction = filters.place_instruction(
                    order_type='MARKET_ON_CLOSE',
                    selection_id=selection_id,
                    side='BACK',
                    market_on_close_order=filters.market_on_close_order(
                        liability=stake
                    )
                )
            
            place_orders = self.trading.betting.place_orders(
                market_id=market_id,
                instructions=[instruction],
                customer_strategy_ref='AlphaEngine'
            )
            
            status = place_orders.status
            if status == 'SUCCESS':
                if hasattr(place_orders, 'place_instruction_reports'):
                    report = place_orders.place_instruction_reports[0]
                elif hasattr(place_orders, 'instruction_reports'):
                    report = place_orders.instruction_reports[0]
                else:
                    return {'status': 'FAILURE', 'error': 'Missing instruction_reports', 'is_success': False}

                return {
                    'status': report.status,
                    'bet_id': report.bet_id,
                    'avg_price_matched': report.average_price_matched,
                    'size_matched': report.size_matched,
                    'is_success': report.status == 'SUCCESS'
                }
            else:
                print(f"Place (back) failure: {place_orders.error_code}")
                if hasattr(place_orders, 'place_instruction_reports') and place_orders.place_instruction_reports:
                    rpt = place_orders.place_instruction_reports[0]
                    print(f"       Inner error: {rpt.error_code}")
                elif hasattr(place_orders, 'instruction_reports') and place_orders.instruction_reports:
                    rpt = place_orders.instruction_reports[0]
                    print(f"       Inner error: {rpt.error_code}")
                    
                return {'status': 'FAILURE', 'error': place_orders.error_code, 'is_success': False}
                
        except Exception as e:
            print(f"Error placing back bet: {e}")
            return {'status': 'ERROR', 'error': str(e), 'is_success': False}

    def cancel_bet(self, market_id: str, bet_id: str) -> bool:
        """
        Cancel a specific bet (order) by its Bet ID.
        """
        try:
            instruction = filters.cancel_instruction(
                bet_id=str(bet_id)
            )
            
            cancel_orders = self.trading.betting.cancel_orders(
                market_id=market_id,
                instructions=[instruction]
            )
            
            # Check top level status
            if cancel_orders.status == 'SUCCESS':
                # Check instruction report for specific bet status
                if hasattr(cancel_orders, 'instruction_reports') and cancel_orders.instruction_reports:
                    report = cancel_orders.instruction_reports[0]
                    if report.status == 'SUCCESS':
                        return True
                    else:
                        print(f"[ERROR] Cancel instruction failed: {report.error_code}")
                        # If error is BET_TAKEN_OR_LAPSED, it means it's already gone (Matched or Expired).
                        # In this case, we can't cancel it, but it IS effectively 'cancelled' from being an active unmatched bet?
                        # No, if it's TAKEN (Matched), we must NOT replace it. Return False.
                        return False
                return True # Default success if no reports?
            
            else:
                print(f"[ERROR] Cancel failed: {cancel_orders.error_code}")
                # Try to dig deeper
                if hasattr(cancel_orders, 'instruction_reports') and cancel_orders.instruction_reports:
                     print(f"       Inner error: {cancel_orders.instruction_reports[0].error_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Cancel exception: {e}")
            return False


def test_betfair():
    """Test Betfair odds fetching"""
    print("="*80)
    print("BETFAIR ODDS FETCHER TEST")
    print("="*80)

    fetcher = BetfairOddsFetcher()

    # Login
    if not fetcher.login():
        print("\n[ERROR] Login failed - cannot continue")
        print("Check your credentials in config.py")
        return

    # Get upcoming markets (next 7 days)
    print("\nFetching upcoming greyhound markets (next 7 days)...")
    from_time = datetime.now()
    to_time = from_time + timedelta(days=7)
    markets = fetcher.get_greyhound_markets(from_time, to_time)
    print(f"[OK] Found {len(markets)} markets")

    if markets:
        print("\nSample markets:")
        for i, market in enumerate(markets[:5], 1):
            event_name = market.event.name if hasattr(market, 'event') else "Unknown"
            print(f"  {i}. {event_name} - {market.market_name}")
            print(f"     Market ID: {market.market_id}")
            print(f"     Start time: {market.market_start_time}")

            # Get odds for first market as example
            if i == 1:
                print(f"\n     Fetching odds for this race...")
                odds = fetcher.get_market_odds(market.market_id)
                if odds:
                    print(f"     Found odds for {len(odds)} runners:")
                    for selection_id, price in list(odds.items())[:5]:
                        print(f"       Selection {selection_id}: ${price:.2f}")

    # Logout
    fetcher.logout()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_betfair()
