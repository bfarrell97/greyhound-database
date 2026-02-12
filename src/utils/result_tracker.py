"""Bet result tracking and CSV logging for live betting.

Maintains a CSV file (live_bets.csv) with bet history, results, and P/L tracking.
Provides methods for logging bets, updating results, and calculating statistics.

Example:
    >>> from src.utils.result_tracker import ResultTracker
    >>> tracker = ResultTracker()
    >>> bet = {
    ...     'MarketID': '1.12345',
    ...     'Dog': 'Fast Freddy',
    ...     'Stake': 10.0,
    ...     'Price': 6.0,
    ...     'BetType': 'BACK'
    ... }
    >>> tracker.log_bet(bet)
"""

import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any


class ResultTracker:
    """Tracks live bets and updates results in CSV file.
    
    Logs all placed bets to 'live_bets.csv' and provides methods to update
    bet status, results (WIN/LOSS), BSP, and profit/loss calculations.
    
    Attributes:
        FILE_PATH (str): Path to CSV file (default: 'live_bets.csv')
        COLUMNS (List[str]): CSV column names
    
    Example:
        >>> tracker = ResultTracker()
        >>> tracker.log_bet({'Dog': 'Fast Freddy', 'Stake': 10.0, ...})
        >>> tracker.update_result('BET123', result='WIN', bsp=5.5, profit=45.0)
    """
    """
    Tracks live bets and updates their results (P&L, Winners, BSP) 
    in a CSV file: 'live_bets.csv'
    """
    
    FILE_PATH: str = "live_bets.csv"
    COLUMNS: List[str] = [
        "MarketID", "SelectionID", "BetID", "Date", "Time", 
        "Track", "Race", "Dog", "BetType", 
        "Status", "Stake", "Price", "BSP", "Result", "Profit"
    ]

    def __init__(self) -> None:
        """Initialize tracker and ensure CSV file exists."""
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.FILE_PATH):
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.FILE_PATH, index=False)

    def log_bet(self, bet_data: Dict[str, Any]) -> None:
        """Log a new bet to the CSV file.
        
        Args:
            bet_data: Dictionary with bet details. Should contain keys matching
                     COLUMNS. Missing keys will be filled with empty strings.
                     Date defaults to today if not provided.
        
        Example:
            >>> bet = {
            ...     'MarketID': '1.12345',
            ...     'Dog': 'Fast Freddy',
            ...     'BetType': 'BACK',
            ...     'Stake': 10.0,
            ...     'Price': 6.0,
            ...     'Status': 'PLACED'
            ... }
            >>> tracker.log_bet(bet)
            [TRACKER] Logged bet: Fast Freddy (PLACED)
        """
        """
        Log a new bet to the CSV.
        bet_data must contain keys matching COLUMNS (or a subset).
        """
        try:
            # Prepare row data
            row = {col: bet_data.get(col, "") for col in self.COLUMNS}
            
            # Use current date if missing
            if not row['Date']:
                row['Date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Default Status if missing
            if not row['Status']:
                row['Status'] = 'PENDING'

            # Append to CSV
            df = pd.DataFrame([row])
            df.to_csv(self.FILE_PATH, mode='a', header=False, index=False)
            print(f"[TRACKER] Logged bet: {row.get('Dog')} ({row.get('Status')})")
            
        except Exception as e:
            print(f"[TRACKER] Error logging bet: {e}")

    def update_results(self, _unused_fetcher=None):
        """
        Poll Betfair for results of PENDING bets and update CSV.
        Manages its OWN session to ensure thread safety.
        """
        if not os.path.exists(self.FILE_PATH):
            return

        # Instantiate LOCAL fetcher for this thread
        from src.integration.betfair_fetcher import BetfairOddsFetcher
        local_fetcher = BetfairOddsFetcher()
        
        try:
            if not local_fetcher.login():
                print("[TRACKER] Failed to login for result update")
                return

            df = pd.read_csv(self.FILE_PATH)
            if df.empty: 
                local_fetcher.logout()
                return

            # Identify PENDING items
            # (Logic unchanged, just using local_fetcher)
            
            # 1. GET SETTLED BETS (For Matched P&L)
            # Fetch ALL statuses (Settled, Voided, Lapsed, Cancelled) with 3-day lookback
            cleared_orders = local_fetcher.get_cleared_orders(bet_status=None, days=3)
            cleared_map = {o.bet_id: o for o in cleared_orders} if cleared_orders else {}

            # 2. ITERATE ROWS
            updated_count = 0
            
            for idx, row in df.iterrows():
                # Skip if already resulted
                if pd.notna(row['Result']) and row['Result'] != "":
                    continue
                    
                market_id = str(row['MarketID'])
                selection_id = int(row['SelectionID']) if pd.notna(row['SelectionID']) and row['SelectionID'] != "" else None
                bet_id = str(row['BetID']) if pd.notna(row['BetID']) else None
                
                # A. CHECK IF MATCHED BET HAS SETTLED
                if bet_id and bet_id in cleared_map:
                    order = cleared_map[bet_id]
                    pnl = order.profit
                    outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "REFUND")
                    
                    df.at[idx, 'Profit'] = round(pnl, 2)
                    df.at[idx, 'Result'] = outcome
                    df.at[idx, 'Status'] = 'SETTLED'
                    updated_count += 1
                
                # B. CHECK MARKET RESULT (For BSP & Unmatched Result)
                if pd.isna(row['BSP']) or row['BSP'] == "":
                    # Fetch Market Result via SINGLE API CALL per market (Optimization could be done here)
                    mkt_res = local_fetcher.get_market_result(market_id)
                    
                    if mkt_res and selection_id in mkt_res:
                        runner_res = mkt_res[selection_id]
                        bsp = runner_res.get('bsp')
                        status = runner_res.get('status') # WINNER / LOSER
                        
                        if bsp:
                            df.at[idx, 'BSP'] = bsp
                        
                        # If result still missing (e.g. Unmatched), log the winner status
                        if pd.isna(row['Result']) or row['Result'] == "":
                            df.at[idx, 'Result'] = status
                            updated_count += 1

            if updated_count > 0:
                df.to_csv(self.FILE_PATH, index=False)
                print(f"[TRACKER] Updated {updated_count} bets with results/BSP")
                
            # Always update summary when checking results
            self.generate_summary()
                
        except Exception as e:
            print(f"[TRACKER] Error updating results: {e}")
        finally:
            local_fetcher.logout()

    def generate_summary(self):
        """Generates a summary report (Daily, Weekly, All-Time)."""
        try:
            if not os.path.exists(self.FILE_PATH): return
            
            # Read CSV
            try:
                df = pd.read_csv(self.FILE_PATH)
            except pd.errors.EmptyDataError:
                return

            if df.empty: return
            
            # Filter only SETTLED or result-containing rows?
            # Or just use Profit column.
            
            # Ensure numeric P/L
            df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
            
            # Timestamp handling: 'Date' is likely just YYYY-MM-DD string.
            # Convert to datetime for proper sorting/week grouping
            df['DateObj'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # 1. Overall Stats
            total_pl = df['Profit'].sum()
            total_bets = len(df)
            win_count = len(df[df['Profit'] > 0])
            strike_rate = (win_count / total_bets * 100) if total_bets > 0 else 0
            
            # 2. Daily Stats
            daily = df.groupby('Date').agg(
                Daily_PL=('Profit', 'sum'),
                Bets=('Profit', 'count'),
                Wins=('Profit', lambda x: (x > 0).sum())
            ).sort_index(ascending=False)
            
            # 3. Weekly Stats
            if df['DateObj'].notna().any():
                df['Week'] = df['DateObj'].dt.to_period('W').astype(str)
                weekly = df.groupby('Week').agg(
                    Weekly_PL=('Profit', 'sum'),
                    Bets=('Profit', 'count')
                ).sort_index(ascending=False)
            else:
                weekly = pd.DataFrame()

            # Write Summary File
            summary_path = self.FILE_PATH.replace('.csv', '_summary.csv')
            
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                f.write("=== ALL TIME PERFORMANCE ===\n")
                f.write(f"Total Profit: ${total_pl:.2f}\n")
                f.write(f"Total Bets: {total_bets}\n")
                f.write(f"Strike Rate: {strike_rate:.1f}%\n")
                f.write("\n")
                
                f.write("=== WEEKLY PERFORMANCE ===\n")
                weekly.to_csv(f, mode='a')
                f.write("\n")
                
                f.write("=== DAILY PERFORMANCE ===\n")
                daily.to_csv(f, mode='a')
                
        except Exception as e:
            print(f"[TRACKER] Error generating summary: {e}")
