"""Bet scheduling system for delayed bet placement.

Schedules bets to be placed at specific times (e.g., T-5 minutes before jump).
Uses background thread to monitor queue and execute bets at scheduled times.

Supports:
- Queue bets for future placement
- Price threshold checks (skip if price too high)
- Status tracking (SCHEDULED, PLACED, SKIPPED, ERROR)
- Callback notifications on bet placement

Example:
    >>> from src.integration.bet_scheduler import BetScheduler
    >>> scheduler = BetScheduler()
    >>> scheduler.start()
    >>> 
    >>> # Schedule bet for 5 minutes before race
    >>> scheduler.schedule_bet(
    ...     bet_id="BET001",
    ...     market_id="1.12345",
    ...     selection_id=789,
    ...     dog_name="Fast Freddy",
    ...     race_time="14:30",
    ...     minutes_before=5,
    ...     stake=10.0,
    ...     max_price=8.0
    ... )
"""
"""
Bet Scheduler Module
====================
Schedules bets to be placed at specific times (e.g., 5 minutes before race start).
Uses a background thread that checks the queue every 10 seconds.
"""
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
import queue


@dataclass
class ScheduledBet:
    """A bet scheduled for future placement.
    
    Attributes:
        bet_id (str): Unique identifier for this scheduled bet
        market_id (str): Betfair MarketID
        selection_id (int): Betfair SelectionID (dog)
        dog_name (str): Dog name
        track_name (str): Track name
        race_number (int): Race number
        race_time (str): Race start time (HH:MM format)
        scheduled_time (datetime): When to place the bet
        stake (float): Bet amount
        max_price (float): Maximum acceptable odds (skip if higher)
        status (str): Current status (SCHEDULED, PLACED, SKIPPED, ERROR)
        result_message (str): Status message or error description
    """
    """A bet scheduled to be placed at a specific time"""
    bet_id: str  # Unique identifier for this scheduled bet
    market_id: str
    selection_id: int
    dog_name: str
    track_name: str
    race_number: int
    race_time: str  # HH:MM format
    scheduled_time: datetime  # When to place the bet
    stake: float
    max_price: float  # Skip if current price > this
    status: str = "SCHEDULED"  # SCHEDULED, PLACED, SKIPPED, ERROR
    result_message: str = ""


class BetScheduler:
    """Background scheduler for delayed bet placement.
    
    Runs in a separate thread, monitoring a queue of scheduled bets and
    executing them at the specified times. Checks price thresholds before
    placing bets and provides status callbacks.
    
    Attributes:
        fetcher: Betfair odds fetcher (for price checks)
        on_bet_placed: Callback function called after bet placement
        _scheduled_bets: Queue of ScheduledBet objects
        _worker_thread: Background worker thread
        _stop_event: Threading event for graceful shutdown
    
    Example:
        >>> def bet_callback(bet):
        ...     print(f"Placed: {bet.dog_name} @ ${bet.stake}")
        >>> 
        >>> scheduler = BetScheduler(on_bet_placed=bet_callback)
        >>> scheduler.start()
        >>> # Schedule bets...
        >>> scheduler.stop()
    """
    """
    Background scheduler that places bets at specified times.
    Designed for placing BACK bets 5 minutes before race start.
    """
    
    def __init__(
        self,
        fetcher: Optional[Any] = None,
        on_bet_placed: Optional[Callable] = None
    ) -> None:
        """Initialize the bet scheduler.

        Args:
            fetcher: Betfair odds fetcher for price checks (optional)
            on_bet_placed: Callback function(bet: ScheduledBet) called after placement
        
        Example:
            >>> scheduler = BetScheduler()
            >>> scheduler.start()
        """
        """
        Initialize the scheduler.

[244 more lines in file. Use offset=82 to continue.]
        
        Args:
            fetcher: BetfairOddsFetcher instance (will create one if None)
            on_bet_placed: Callback function(bet_id, status, message) when bet is processed
        """
        self.fetcher = fetcher
        self.on_bet_placed = on_bet_placed
        self.scheduled_bets: Dict[str, ScheduledBet] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._bet_counter = 0
        
    def start(self):
        """Start the background scheduler thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # print("[BetScheduler] Started background scheduler")
        
    def stop(self):
        """Stop the background scheduler thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("[BetScheduler] Stopped")
        
    def schedule_bet(self, market_id: str, selection_id: int, dog_name: str,
                     track_name: str, race_number: int, race_time: str,
                     stake: float = 1.0, max_price: float = 8.0,
                     minutes_before: int = 5, allow_duplicate: bool = False) -> str:
        """
        Schedule a bet to be placed X minutes before race time.
        
        Args:
            market_id: Betfair market ID
            selection_id: Runner selection ID
            dog_name: Dog name
            track_name: Track name
            race_number: Race number
            race_time: Race time in HH:MM format
            stake: Stake amount (default $1)
            max_price: Maximum price to place bet (skip if higher)
            minutes_before: Minutes before race to place bet (default 5)
            allow_duplicate: If True allow multiple scheduled bets for same selection
            
        Returns:
            Unique bet_id for tracking (returns existing bet_id if duplicate found and allow_duplicate=False)
        """
        # If not allowed, check for existing scheduled bet for same market+selection
        if not allow_duplicate:
            with self.lock:
                for existing in self.scheduled_bets.values():
                    if (existing.market_id == market_id and
                        existing.selection_id == selection_id and
                        existing.status == "SCHEDULED"):
                        print(f"[BetScheduler] Duplicate schedule detected for {dog_name} - returning existing bet {existing.bet_id}")
                        return existing.bet_id

        # Generate unique bet ID
        self._bet_counter += 1
        bet_id = f"SCHED_{datetime.now().strftime('%H%M%S')}_{self._bet_counter}"
        
        # Calculate scheduled time
        today = datetime.now().date()
        try:
            race_dt = datetime.strptime(f"{today} {race_time}", "%Y-%m-%d %H:%M")
        except:
            race_dt = datetime.strptime(f"{today} {race_time[:5]}", "%Y-%m-%d %H:%M")
            
        scheduled_time = race_dt - timedelta(minutes=minutes_before)
        
        bet = ScheduledBet(
            bet_id=bet_id,
            market_id=market_id,
            selection_id=selection_id,
            dog_name=dog_name,
            track_name=track_name,
            race_number=race_number,
            race_time=race_time,
            scheduled_time=scheduled_time,
            stake=stake,
            max_price=max_price
        )
        
        with self.lock:
            self.scheduled_bets[bet_id] = bet
            
        print(f"[BetScheduler] Scheduled: {dog_name} @ {race_time} -> place at {scheduled_time.strftime('%H:%M:%S')}")
        return bet_id
        
    def cancel_bet(self, bet_id: str) -> bool:
        """Cancel a scheduled bet"""
        with self.lock:
            if bet_id in self.scheduled_bets:
                bet = self.scheduled_bets[bet_id]
                if bet.status == "SCHEDULED":
                    bet.status = "CANCELLED"
                    return True
        return False
        
    def get_scheduled_bets(self) -> List[ScheduledBet]:
        """Get list of all scheduled bets"""
        with self.lock:
            return list(self.scheduled_bets.values())
            
    def get_bet_status(self, bet_id: str) -> Optional[ScheduledBet]:
        """Get status of a specific bet"""
        with self.lock:
            return self.scheduled_bets.get(bet_id)
            
    def _run_loop(self):
        """Background loop that checks for due bets"""
        # print("[BetScheduler] Background loop started")
        
        while self.running:
            try:
                self._check_and_place_bets()
            except Exception as e:
                print(f"[BetScheduler] Error in loop: {e}")
            
            # Sleep 10 seconds between checks
            time.sleep(10)
            
    def _check_and_place_bets(self):
        """Check if any bets are due and place them"""
        now = datetime.now()
        
        with self.lock:
            pending = [b for b in self.scheduled_bets.values() if b.status == "SCHEDULED"]
            
        for bet in pending:
            if now >= bet.scheduled_time:
                self._place_bet(bet)
                
    def _place_bet(self, bet: ScheduledBet):
        """Actually place the bet via Betfair API"""
        print(f"[BetScheduler] Placing bet: {bet.dog_name} ({bet.track_name} R{bet.race_number})")
        
        # Initialize fetcher if needed
        if not self.fetcher:
            try:
                from src.integration.betfair_fetcher import BetfairOddsFetcher
                self.fetcher = BetfairOddsFetcher()
                if not self.fetcher.login():
                    self._update_bet_status(bet, "ERROR", "Betfair login failed")
                    return
            except Exception as e:
                self._update_bet_status(bet, "ERROR", f"Init error: {e}")
                return
        
        # Check for existing active orders for this selection (prevent duplicate placement)
        try:
            if self._has_active_order(bet.market_id, bet.selection_id, side='BACK'):
                self._update_bet_status(bet, "SKIPPED", "Existing active order for this selection")
                return
        except Exception as e:
            print(f"[BetScheduler] Warning: failed to check active orders: {e}")
        
        try:
            # Get current back price
            odds = self.fetcher.get_market_odds(bet.market_id)
            current_price = odds.get(bet.selection_id)
            
            if not current_price:
                self._update_bet_status(bet, "SKIPPED", "No price available")
                return
                
            # Check max price limit
            if current_price > bet.max_price:
                self._update_bet_status(bet, "SKIPPED", f"Price ${current_price:.2f} > max ${bet.max_price:.2f}")
                return
                
            # Place the back bet
            result = self.fetcher.place_back_bet(
                market_id=bet.market_id,
                selection_id=bet.selection_id,
                stake=bet.stake,
                price=current_price
            )
            
            if result.get('is_success'):
                msg = f"PLACED @ ${current_price:.2f} (Bet ID: {result.get('bet_id')})"
                self._update_bet_status(bet, "PLACED", msg)
            else:
                error = result.get('error', 'Unknown error')
                self._update_bet_status(bet, "ERROR", f"Failed: {error}")
                
        except Exception as e:
            self._update_bet_status(bet, "ERROR", f"Exception: {e}")
            
    def _has_active_order(self, market_id: str, selection_id: int, side: str = 'BACK') -> bool:
        """Return True if an active (unmatched/partially matched/open) order exists for the runner"""
        try:
            if not self.fetcher:
                return False
            current_orders = self.fetcher.get_current_orders(market_id=market_id)
            for o in current_orders:
                # Try different attribute names from Betfair objects
                sel = getattr(o, 'selection_id', None) or getattr(o, 'selectionId', None) or getattr(o, 'selectionId', None)
                o_side = getattr(o, 'side', None)
                status = getattr(o, 'status', None)
                size_unmatched = getattr(o, 'size_remaining', None) or getattr(o, 'size_unmatched', None) or getattr(o, 'size_remaining', None)

                try:
                    sel = int(sel) if sel is not None else None
                except:
                    sel = None

                if sel == selection_id:
                    if o_side and o_side.upper() != side.upper():
                        continue
                    # If there's any remaining unmatched size or status indicates open, treat as active
                    if size_unmatched and float(size_unmatched) > 0:
                        return True
                    if status and status.upper() in ('EXECUTABLE', 'EXECUTED', 'PARTIALLY_MATCHED', 'PENDING'):
                        return True
            return False
        except Exception as e:
            print(f"[BetScheduler] Error checking active orders: {e}")
            return False

    def _update_bet_status(self, bet: ScheduledBet, status: str, message: str):
        """Update bet status and notify callback"""
        with self.lock:
            bet.status = status
            bet.result_message = message
            
        print(f"[BetScheduler] {bet.dog_name}: {status} - {message}")
        
        if self.on_bet_placed:
            try:
                self.on_bet_placed(bet.bet_id, status, message)
            except Exception as e:
                print(f"[BetScheduler] Callback error: {e}")


# Global scheduler instance
_scheduler: Optional[BetScheduler] = None

def get_scheduler() -> BetScheduler:
    """Get or create the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = BetScheduler()
        _scheduler.start()
    return _scheduler


if __name__ == "__main__":
    # Test the scheduler
    print("Testing BetScheduler...")
    
    scheduler = BetScheduler()
    scheduler.start()
    
    # Schedule a test bet 10 seconds from now
    now = datetime.now()
    test_time = (now + timedelta(minutes=5, seconds=15)).strftime("%H:%M")
    
    bet_id = scheduler.schedule_bet(
        market_id="TEST_MARKET",
        selection_id=12345,
        dog_name="TEST DOG",
        track_name="TEST TRACK",
        race_number=1,
        race_time=test_time,
        stake=1.0,
        max_price=8.0,
        minutes_before=5
    )
    
    print(f"Scheduled bet: {bet_id}")
    print("Waiting 30 seconds...")
    time.sleep(30)
    
    status = scheduler.get_bet_status(bet_id)
    print(f"Status: {status}")
    
    scheduler.stop()
