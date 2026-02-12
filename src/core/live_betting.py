"""Live betting manager for production Betfair API integration.

Handles real-money bet placement, tracking, and P/L calculation via Betfair Exchange.
Maintains LiveBets table in the main database for persistent bet tracking.

Features:
- Place and track bets via Betfair API
- Update bet status from Betfair current orders
- Settle bets and calculate P/L with commission
- Generate daily, weekly, and all-time statistics
- Prevent duplicate bets (same dog, race, today)

Example:
    >>> from src.core.live_betting import LiveBettingManager
    >>> manager = LiveBettingManager()
    >>> manager.place_bet_record(
    ...     bet_id='123456',
    ...     market_id='1.12345',
    ...     selection_id=789,
    ...     price=6.0,
    ...     size=10.0,
    ...     side='BACK',
    ...     market_name='R5 Wentworth Park',
    ...     selection_name='Fast Freddy'
    ... )
"""

import sqlite3
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
import threading
import pandas as pd


class LiveBettingManager:
    """Manager for live (real-money) betting via Betfair Exchange.
    
    Tracks bets in LiveBets database table. Syncs with Betfair API for bet status
    updates and settlement. Calculates P/L with 5% commission on winnings.
    
    Attributes:
        db_path (str): Path to SQLite database
    
    Example:
        >>> manager = LiveBettingManager()
        >>> active = manager.get_active_bets()
        >>> print(f"Active bets: {len(active)}")
        Active bets: 3
    """

    def __init__(self, db_path: str = 'greyhound_racing.db') -> None:
        """Initialize live betting manager and create database tables.
        
        Args:
            db_path: Path to SQLite database. Defaults to 'greyhound_racing.db'.
        """
        self.db_path = db_path
        self._init_db()
        
    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection.
        
        Returns:
            SQLite connection object
        """
        return sqlite3.connect(self.db_path)
        
    def _init_db(self) -> None:
        """Initialize LiveBets table and perform schema migrations if needed.
        
        Creates table with columns: BetID, MarketID, SelectionID, PlacedDate,
        Price, Size, Side, Status, Result, Profit, Commission, MarketName,
        SelectionName, MeetingDate, RaceTime (optional), Strategy (optional).
        """
        """Initialize LiveBets table"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LiveBets (
                BetID TEXT PRIMARY KEY,
                MarketID TEXT,
                SelectionID INTEGER,
                PlacedDate TIMESTAMP,
                Price REAL,
                Size REAL,
                Side TEXT,
                Status TEXT, -- 'PLACED', 'MATCHED', 'SETTLED', 'LAPSED'
                Result TEXT, -- 'WIN', 'LOSE', 'ERA'
                Profit REAL,
                Commission REAL,
                MarketName TEXT,
                SelectionName TEXT,
                MeetingDate DATE
            )
        """)

        
        # Check for missing columns (Schema Migration)
        cursor.execute("PRAGMA table_info(LiveBets)")
        columns = [info[1] for info in cursor.fetchall()]
        
        missing_cols = {
            'Result': 'TEXT',
            'Profit': 'REAL',
            'Commission': 'REAL',
            'MarketName': 'TEXT',
            'SelectionName': 'TEXT',
            'MeetingDate': 'DATE',
            'RaceTime': 'TEXT',
            'Strategy': 'TEXT'
        }
        
        for col, dtype in missing_cols.items():
            if col not in columns:
                print(f"[INFO] Migrating database: Adding column {col}")
                try:
                    cursor.execute(f"ALTER TABLE LiveBets ADD COLUMN {col} {dtype}")
                except Exception as e:
                    print(f"[WARN] Failed to add column {col}: {e}")

        conn.commit()
        conn.close()

    def place_bet_record(
        self,
        bet_id: str,
        market_id: str,
        selection_id: int,
        price: float,
        size: float,
        side: str,
        market_name: str,
        selection_name: str,
        race_time: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> bool:
        """Record a newly placed bet in the database.
        
        Args:
            bet_id: Betfair BetID (unique identifier from API)
            market_id: Betfair MarketID (e.g., '1.12345')
            selection_id: Betfair SelectionID (runner identifier)
            price: Odds (decimal format, e.g., 6.0)
            size: Stake amount in currency units
            side: 'BACK' or 'LAY'
            market_name: Race description (e.g., 'R5 Wentworth Park')
            selection_name: Dog name
            race_time: Race start time (optional, format: 'HH:MM:SS')
            strategy: Strategy used (optional, e.g., 'V44_BACK')
        
        Returns:
            True if bet recorded successfully, False on error
        
        Example:
            >>> manager.place_bet_record(
            ...     bet_id='123456789',
            ...     market_id='1.234567',
            ...     selection_id=789,
            ...     price=6.0,
            ...     size=10.0,
            ...     side='BACK',
            ...     market_name='R5 Wentworth Park 14:30',
            ...     selection_name='Fast Freddy',
            ...     strategy='V44_STEAMER'
            ... )
            True
        """
        """Record a newly placed bet"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO LiveBets 
                (BetID, MarketID, SelectionID, PlacedDate, Price, Size, Side, Status, MarketName, SelectionName, MeetingDate, RaceTime, Strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'MATCHED', ?, ?, ?, ?, ?)
            """, (
                str(bet_id), market_id, selection_id, datetime.now(), 
                price, size, side, market_name, selection_name,
                datetime.now().date(), race_time, strategy
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error recording bet: {e}")
            return False
        finally:
            conn.close()

    def get_placed_dogs_today(self) -> set:
        """Get set of dog names with bets placed today (prevents duplicates).
        
        Used to check if a dog already has a bet before placing a new one.
        
        Returns:
            Set of dog names (str) with bets placed today
        
        Example:
            >>> placed_today = manager.get_placed_dogs_today()
            >>> if 'Fast Freddy' in placed_today:
            ...     print("Already bet on this dog today")
            Already bet on this dog today
        """
        """Get set of dog names with bets placed today"""
        conn = self._get_conn()
        cursor = conn.cursor()
        today = datetime.now().date()
        
        try:
            # Check for bets placed today that aren't void/lapsed
            cursor.execute("""
                SELECT SelectionName FROM LiveBets 
                WHERE MeetingDate = ? 
                  AND Status NOT IN ('LAPSED', 'VOID')
            """, (today,))
            
            dogs = set()
            import re
            for row in cursor.fetchall():
                if row[0]:
                    name = str(row[0]).strip()
                    # Remove trap number prefix like "8. " from "8. Frank Grimes"
                    name = re.sub(r'^\d+\.\s*', '', name)
                    dogs.add(name.upper())
            
            print(f"[DEBUG] Placed dogs today: {dogs}")
            return dogs
        except Exception as e:
            print(f"Error checking placed dogs: {e}")
            return set()
        finally:
            conn.close()

    def update_from_betfair_orders(self, current_orders, cleared_orders, fetcher=None):
        """Update local DB from Betfair order streams"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Debug: Print first order structure (disable in prod)
        # if current_orders and len(current_orders) > 0:
        #     first = current_orders[0]
        #     print(f"[DEBUG] First order type: {type(first)}")
        
        # 1. Process Current Orders (Active/Pending)
        for order in current_orders:
            # Handle both object and dict styles
            bet_id = getattr(order, 'bet_id', None) or (order.get('betId') if hasattr(order, 'get') else None)
            market_id = getattr(order, 'market_id', None) or (order.get('marketId') if hasattr(order, 'get') else None)
            selection_id = getattr(order, 'selection_id', None) or (order.get('selectionId') if hasattr(order, 'get') else None)
            price = getattr(order, 'average_price_matched', None) or getattr(order, 'price', None) or 0
            size = getattr(order, 'size_matched', None) or getattr(order, 'size', None) or 0
            side = getattr(order, 'side', None) or (order.get('side', 'BACK') if hasattr(order, 'get') else 'BACK')
            status = getattr(order, 'status', None) or (order.get('status', 'MATCHED') if hasattr(order, 'get') else 'MATCHED')
            placed_date = getattr(order, 'placed_date', None) or (order.get('placedDate', datetime.now()) if hasattr(order, 'get') else datetime.now())
            
            # Get enriched names (attached by refresh_live_orders)
            market_name = getattr(order, '_enriched_market_name', None)
            runner_name = getattr(order, '_enriched_runner_name', None)
            race_time = getattr(order, '_enriched_race_time', None)
            
            if not bet_id:
                continue
                
            # Check if exists
            cursor.execute("SELECT BetID FROM LiveBets WHERE BetID = ?", (str(bet_id),))
            exists = cursor.fetchone()
            
            if exists:
                # Update status and names if we have them
                if market_name or runner_name or race_time:
                    cursor.execute("""
                        UPDATE LiveBets SET Status = ?, Price = ?, Size = ?, 
                        MarketName = COALESCE(?, MarketName), 
                        SelectionName = COALESCE(?, SelectionName),
                        RaceTime = COALESCE(?, RaceTime)
                        WHERE BetID = ?
                    """, (status, price, size, market_name, runner_name, race_time, str(bet_id)))
                else:
                    cursor.execute("""
                        UPDATE LiveBets SET Status = ?, Price = ?, Size = ?
                        WHERE BetID = ?
                    """, (status, price, size, str(bet_id)))
            else:
                # Insert new with names
                cursor.execute("""
                    INSERT INTO LiveBets 
                    (BetID, MarketID, SelectionID, PlacedDate, Price, Size, Side, Status, MeetingDate, MarketName, SelectionName, RaceTime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(bet_id), market_id, selection_id, placed_date,
                    price, size, side, status, datetime.now().date(),
                    market_name, runner_name, race_time
                ))
        
        # 2. Update Cleared Orders (Settled)
        for order in cleared_orders:
            bet_id = getattr(order, 'bet_id', None) or order.get('betId')
            profit = getattr(order, 'profit', None) or order.get('profit', 0)
            status = 'SETTLED'
            result = 'WIN' if profit > 0 else 'LOSE'
            if profit == 0: result = 'ERA' # void/push
            
            cursor.execute("""
                UPDATE LiveBets 
                SET Status = ?, Result = ?, Profit = ?
                WHERE BetID = ?
            """, (status, result, profit, str(bet_id)))
            
            # If bet wasn't in DB, insert it (captured from external source/website)
            if cursor.rowcount == 0:
                # Extract details safely
                market_id = getattr(order, 'market_id', None) or order.get('marketId')
                selection_id = getattr(order, 'selection_id', None) or order.get('selectionId')
                placed_date = getattr(order, 'placed_date', datetime.now()) or order.get('placedDate', datetime.now())
                price = getattr(order, 'price_matched', 0) or order.get('priceMatched', 0)
                size = getattr(order, 'size_settled', 0) or order.get('sizeSettled', 0)
                side = getattr(order, 'side', 'BACK') or order.get('side', 'BACK')
                
                # Try to get item description if available (sometimes nested)
                market_name = "External Bet"
                selection_name = f"Selection {selection_id}"
                
                # Check for itemDescription
                if hasattr(order, 'item_description'):
                    desc = order.item_description
                    if desc:
                        if hasattr(desc, 'market_desc'): market_name = desc.market_desc
                        if hasattr(desc, 'runner_desc'): selection_name = desc.runner_desc
                
                cursor.execute("""
                    INSERT INTO LiveBets 
                    (BetID, MarketID, SelectionID, PlacedDate, Price, Size, Side, Status, Result, Profit, MeetingDate, MarketName, SelectionName)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(bet_id), market_id, selection_id, placed_date,
                    price, size, side, status, result, profit,
                    datetime.now().date(), market_name, selection_name
                )) 

        conn.commit()
        conn.close()

    def get_active_bets(self) -> pd.DataFrame:
        """Get today's active bets (placed, matched, or executing).
        
        Returns bets with status PLACED, MATCHED, EXECUTION_COMPLETE, or EXECUTABLE
        that have today's meeting date. Sorted by race time ascending.
        
        Returns:
            DataFrame with active bet records
        
        Example:
            >>> active = manager.get_active_bets()
            >>> print(f"{len(active)} active bets")
            3 active bets
        """
        """Get today's unmatched or matched but unfinished bets"""
        conn = self._get_conn()
        today = datetime.now().date()
        # Include EXECUTION_COMPLETE (Betfair's "fully matched" status) - but only for TODAY
        query = """
            SELECT * FROM LiveBets 
            WHERE Status IN ('PLACED', 'MATCHED', 'EXECUTION_COMPLETE', 'EXECUTABLE') 
              AND MeetingDate = ?
            ORDER BY RaceTime ASC
        """
        df = pd.read_sql_query(query, conn, params=(today,))
        conn.close()
        return df

    def get_settled_bets(self, limit: int = 50) -> pd.DataFrame:
        """Get recently settled bets (completed and finalized).
        
        Args:
            limit: Maximum number of bets to return. Defaults to 50.
        
        Returns:
            DataFrame with settled bet records, newest first
        
        Example:
            >>> settled = manager.get_settled_bets(limit=10)
            >>> wins = settled[settled['Result'] == 'WIN']
            >>> print(f"{len(wins)}/10 winners")
            6/10 winners
        """
        """Get history"""
        conn = self._get_conn()
        query = f"SELECT * FROM LiveBets WHERE Status = 'SETTLED' ORDER BY PlacedDate DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_todays_summary(self) -> Dict[str, Any]:
        """Get today's betting performance summary.
        
        Calculates total bets, wins, losses, strike rate, and P/L for today.
        
        Returns:
            Dictionary with keys: bets, wins, losses, strike_rate, profit, turnover
        
        Example:
            >>> summary = manager.get_todays_summary()
            >>> print(f"Today: {summary['wins']}/{summary['bets']} (+${summary['profit']:.2f})")
            Today: 5/12 (+$15.30)
        """
        """Get P/L for today"""
        conn = self._get_conn()
        today = datetime.now().date()
        query = """
            SELECT 
                COUNT(*) as count,
                SUM(CASE WHEN Result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(Profit) as profit,
                SUM(Size) as turnover
            FROM LiveBets 
            WHERE Status='SETTLED' AND MeetingDate = ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (today,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'bets': row[0] or 0,
                'wins': row[1] or 0,
                'profit': row[2] or 0.0,
                'turnover': row[3] or 0.0
            }
        return {'bets': 0, 'wins': 0, 'profit': 0.0, 'turnover': 0.0}

    def get_weekly_stats(self):
        """Get P/L for last 7 days"""
        conn = self._get_conn()
        from datetime import date, timedelta
        seven_days_ago = date.today() - timedelta(days=7)
        query = """
            SELECT 
                COUNT(*) as count,
                SUM(CASE WHEN Result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(Profit) as profit,
                SUM(Size) as turnover
            FROM LiveBets 
            WHERE Status='SETTLED' AND MeetingDate >= ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (seven_days_ago,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'bets': row[0] or 0,
                'wins': row[1] or 0,
                'profit': row[2] or 0.0,
                'turnover': row[3] or 0.0
            }
        return {'bets': 0, 'wins': 0, 'profit': 0.0, 'turnover': 0.0}

    def get_strategy_stats(self):
        """Get ROI grouped by Strategy"""
        conn = self._get_conn()
        query = """
            SELECT 
                Strategy,
                COUNT(*) as count,
                SUM(CASE WHEN Result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(Profit) as profit,
                SUM(Size) as turnover
            FROM LiveBets 
            WHERE Status='SETTLED'
            GROUP BY Strategy
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_all_time_stats(self):
        """Get P/L for all time"""
        conn = self._get_conn()
        query = """
            SELECT 
                COUNT(*) as count,
                SUM(CASE WHEN Result='WIN' THEN 1 ELSE 0 END) as wins,
                SUM(Profit) as profit,
                SUM(Size) as turnover
            FROM LiveBets 
            WHERE Status='SETTLED'
        """
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'bets': row[0] or 0,
                'wins': row[1] or 0,
                'profit': row[2] or 0.0,
                'turnover': row[3] or 0.0
            }
        return {'bets': 0, 'wins': 0, 'profit': 0.0, 'turnover': 0.0}

    def clear_database(self):
        """Reset history - User requested feature"""
        conn = self._get_conn()
        conn.execute("DELETE FROM LiveBets")
        conn.commit()
        conn.close()

    def backfill_race_times(self):
        """Backfill missing RaceTime for active bets by querying Races table"""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Find bets with missing time
            cursor.execute("SELECT BetID, MarketName, MeetingDate FROM LiveBets WHERE (RaceTime IS NULL OR RaceTime = '') AND Status IN ('PLACED', 'MATCHED')")
            rows = cursor.fetchall()
            
            updates = 0
            import re
            
            for bet_id, market_name, date_str in rows:
                if not market_name: continue
                
                # Parse "Track R#" e.g. "Dubbo R4"
                match = re.search(r'(.+) R(\d+)', market_name)
                if match:
                    track = match.group(1).strip()
                    race_num = match.group(2)
                    
                    # Query Races table with JOINs
                    query = """
                        SELECT r.RaceTime 
                        FROM Races r
                        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                        JOIN Tracks t ON rm.TrackID = t.TrackID
                        WHERE t.TrackName = ? AND r.RaceNumber = ? AND rm.MeetingDate = ?
                    """
                    cursor.execute(query, (track, race_num, date_str))
                    res = cursor.fetchone()
                    if res and res[0]:
                        time_val = res[0]
                        cursor.execute("UPDATE LiveBets SET RaceTime = ? WHERE BetID = ?", (time_val, bet_id))
                        updates += 1
            
            if updates > 0:
                print(f"[INFO] Backfilled RaceTime for {updates} bets.")
            conn.commit()
        except Exception as e:
            print(f"Backfill error: {e}")
        finally:
            conn.close()
