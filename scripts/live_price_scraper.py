"""
Live Price Scraper Service
===========================
Monitors upcoming Betfair races and captures live prices at specific time
intervals before race start. Stores prices in the database for model training.

Price Capture Schedule:
- 60 minutes before: Price60Min
- 30 minutes before: Price30Min  
- 15 minutes before: Price15Min
- 10 minutes before: Price10Min
- 5 minutes before: Price5Min
- 2 minutes before: Price2Min
- 1 minute before: Price1Min
- After race settles: BSP (Betfair Starting Price)

Usage:
    python scripts/live_price_scraper.py
    
Keep running in background to continuously capture prices.
"""
import sqlite3
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.betfair_fetcher import BetfairOddsFetcher

# Price capture points (minutes before race -> column name)
# 0 means BSP (captured after race)
PRICE_INTERVALS = {
    60: 'Price60Min',
    30: 'Price30Min',
    15: 'Price15Min',
    10: 'Price10Min',
    5: 'Price5Min',
    2: 'Price2Min',
    1: 'Price1Min',
}

# BSP is captured separately after race settles
BSP_COLUMN = 'BSP'

# How often to check for races to scrape (seconds) - Reduced from 30 to 15 for faster monitoring
CHECK_INTERVAL = 15

# Database path
DB_PATH = 'greyhound_racing.db'


class LivePriceScraper:
    """Background service to capture live Betfair prices"""
    
    def __init__(self):
        self.fetcher = BetfairOddsFetcher()
        self.logged_in = False
        # Track which prices we've already captured: {(market_id, selection_id, interval): True}
        self.captured = {}
        
    def start(self):
        """Start the scraper service"""
        print("="*60)
        print("LIVE PRICE SCRAPER SERVICE")
        print("="*60)
        print(f"Capture intervals: {list(PRICE_INTERVALS.keys())} mins before race")
        print(f"Check frequency: Every {CHECK_INTERVAL} seconds")
        print("-"*60)
        
        # Login to Betfair
        if not self.fetcher.login():
            print("[ERROR] Failed to login to Betfair")
            return
        self.logged_in = True
        print("[OK] Logged in to Betfair")
        
        try:
            while True:
                self._check_and_capture()
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n[STOP] Scraper stopped by user")
        finally:
            self.fetcher.logout()
            
    def _check_and_capture(self):
        """Check upcoming races and capture prices as needed"""
        now = datetime.utcnow()  # Use UTC to match Betfair API
        
        # Get greyhound markets for next 2 hours (WIN ONLY to avoid Place price contamination)
        markets = self.fetcher.get_greyhound_markets(
            from_time=now,
            to_time=now + timedelta(hours=2),
            market_type_codes=['WIN']  # CRITICAL: Only fetch Win markets
        )
        
        if not markets:
            return
        
        # DEBUG: Log market types found
        win_count = 0
        place_count = 0
        place_names = []
        for m in markets:
            mn = (m.market_name or '').lower()
            if 'to be placed' in mn or 'tbp' in mn or 'place' in mn:
                place_count += 1
                if len(place_names) < 3:
                    place_names.append(m.market_name)
            else:
                win_count += 1
        print(f"[DEBUG] Markets breakdown: {win_count} WIN, {place_count} PLACE")
        if place_names:
            print(f"[DEBUG] Sample PLACE markets: {place_names}")
            
        for market in markets:
            try:
                self._process_market(market, now)
            except Exception as e:
                print(f"[ERROR] Market {market.market_id}: {e}")
        
        # Also check for recently settled races to capture BSP
        self._capture_bsp_for_settled_races(now)
                
    def _capture_bsp_for_settled_races(self, now: datetime):
        """Capture BSP for races that have just settled"""
        # Get markets from past 30 mins to 2 hours ago (should be settled)
        try:
            markets = self.fetcher.get_greyhound_markets(
                from_time=now - timedelta(hours=2),
                to_time=now - timedelta(minutes=5),
                market_type_codes=['WIN']  # WIN only for consistency
            )
        except:
            return
            
        if not markets:
            return
            
        for market in markets:
            try:
                self._capture_bsp(market)
            except Exception as e:
                pass  # Silently fail for BSP - race may not be settled yet
                
    def _capture_bsp(self, market):
        """Capture BSP for a settled market"""
        market_id = market.market_id
        
        # Check if already captured BSP for this market
        bsp_key = (market_id, 'BSP')
        if bsp_key in self.captured:
            return
            
        # Get market book with SP data
        try:
            market_books = self.fetcher.trading.betting.list_market_book(
                market_ids=[market_id],
                price_projection={'priceData': ['SP_TRADED']}
            )
        except:
            return
            
        if not market_books or not market_books[0].runners:
            return
            
        # Parse market info
        track_name, race_number = self._parse_market_name(market)
        if not track_name or not race_number:
            return
            
        race_date = market.market_start_time.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(DB_PATH)
        updated = 0
        
        try:
            for runner in market_books[0].runners:
                # Get actual SP (BSP)
                if not hasattr(runner, 'sp') or not runner.sp:
                    continue
                bsp = getattr(runner.sp, 'actual_sp', None)
                if not bsp or bsp <= 0:
                    continue
                    
                # Get dog name
                dog_name = self._get_runner_name(market, runner.selection_id)
                if not dog_name:
                    continue
                    
                # Update database
                cursor = conn.execute("""
                    UPDATE GreyhoundEntries 
                    SET BSP = ?
                    WHERE RaceID IN (
                        SELECT r.RaceID FROM Races r
                        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                        JOIN Tracks t ON rm.TrackID = t.TrackID
                        WHERE t.TrackName LIKE ? 
                        AND rm.MeetingDate = ?
                        AND r.RaceNumber = ?
                    )
                    AND GreyhoundID IN (
                        SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName LIKE ?
                    )
                """, (bsp, f"%{track_name}%", race_date, race_number, f"%{dog_name}%"))
                
                if cursor.rowcount > 0:
                    updated += 1
                    
            conn.commit()
            
            if updated > 0:
                self.captured[bsp_key] = True
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {track_name} R{race_number}: "
                      f"Captured {updated} BSP values")
                      
        except Exception as e:
            print(f"[ERROR] BSP update failed: {e}")
        finally:
            conn.close()
                
    def _process_market(self, market, now: datetime):
        """Process a single market for price capture"""
        # Get race start time
        race_time = market.market_start_time
        if race_time.tzinfo:
            race_time = race_time.replace(tzinfo=None)
            
        # Calculate time to race
        time_to_race = (race_time - now).total_seconds() / 60  # minutes

        # DEBUG: Print time diff for first race found to verify calc
        # print(f"[DEBUG] {market.market_name}: Race Time {race_time} (UTC), Now {now} (UTC), Diff {time_to_race:.1f} mins")

        
        # Check each interval
        # Check each interval
        for interval_mins, base_column_name in PRICE_INTERVALS.items():
            # Capture window: interval +/- 1 minute
            if interval_mins - 1 <= time_to_race <= interval_mins + 1:
                
                # Determine market type from market_name (avoid MARKET_DESCRIPTION which causes TOO_MUCH_DATA)
                # ROBUST IDENTIFICATION using Market Description (if available)
                is_place_market = False
                
                if hasattr(market, 'description') and market.description:
                    if getattr(market.description, 'market_type', '') == 'PLACE':
                        is_place_market = True
                else:
                    # Fallback to name parsing
                    market_name = market.market_name or ''
                    market_name_lower = market_name.lower()
                    if 'to be placed' in market_name_lower or 'tbp' in market_name_lower or 'place' in market_name_lower:
                        is_place_market = True
                
                if is_place_market:
                    final_column = 'Place' + base_column_name
                else:
                    # Assume WIN market
                    final_column = base_column_name
                    
                self._capture_prices(market, final_column, interval_mins)
                
    def _ensure_race_in_db(self, market):
        """Ensure market exists in DB, insert if missing (Self-Healing Schedule)"""
        try:
            # Parse Market Info
            track_name, race_number = self._parse_market_name(market)
            if not track_name or not race_number: 
                return False
                
            # Race Date
            race_time_dt = market.market_start_time
            if race_time_dt.tzinfo:
                 race_time_dt = race_time_dt.replace(tzinfo=None) # UTC naive
            
            # Simple Date Str (local/UTC mix is tricky, standardizing on market start date)
            # In live scraper, market.market_start_time is usually UTC. 
            # Our DB expects YYYY-MM-DD.
            race_date = race_time_dt.strftime('%Y-%m-%d')
            
            # Distance (Optional)
            import re
            dist_match = re.search(r'(\d+)m', market.market_name)
            distance = int(dist_match.group(1)) if dist_match else 0
            
            race_time_str = race_time_dt.strftime('%H:%M') # Simplified UTC-ish time for ID
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # 1. Track
            c.execute("SELECT TrackID FROM Tracks WHERE TrackName = ?", (track_name,))
            row = c.fetchone()
            if row:
                track_id = row[0]
            else:
                # Generate TrackKey (Required NOT NULL)
                track_key = track_name.lower().replace(' ', '-')
                c.execute("INSERT INTO Tracks (TrackKey, TrackName, State) VALUES (?, ?, ?)", (track_key, track_name, 'UNK'))
                track_id = c.lastrowid
                print(f"[AUTO-DB] Created Track: {track_name} (Key: {track_key})")
                
            # 2. Meeting
            c.execute("SELECT MeetingID FROM RaceMeetings WHERE TrackID = ? AND MeetingDate = ?", (track_id, race_date))
            row = c.fetchone()
            if row:
                meeting_id = row[0]
            else:
                c.execute("INSERT INTO RaceMeetings (TrackID, MeetingDate) VALUES (?, ?)", (track_id, race_date))
                meeting_id = c.lastrowid
                
            # 3. Race
            c.execute("SELECT RaceID FROM Races WHERE MeetingID = ? AND RaceNumber = ?", (meeting_id, race_number))
            row = c.fetchone()
            if row:
                race_id = row[0]
            else:
                c.execute("INSERT INTO Races (MeetingID, RaceNumber, RaceTime, Distance, Grade) VALUES (?, ?, ?, ?, ?)",
                          (meeting_id, race_number, race_time_str, distance, 'Mixed'))
                race_id = c.lastrowid
                print(f"[AUTO-DB] Created Race: {track_name} R{race_number}")

            # 4. Dogs (Only if we just created the race or to be safe)
            # To avoid slam, only check dogs if we created the race or 1/100 chance?
            # Actually, we need dogs to capture prices. So check dogs always if price capture is imminent.
            
            if market.runners:
                for runner in market.runners:
                    d_name = runner.runner_name
                    d_name = re.sub(r'^\d+\.\s*', '', d_name).strip()
                    d_name = d_name.replace('(Res)', '').strip()
                    d_name = d_name.split('(')[0].strip().upper()
                    
                    # Robust Box extraction (Prevent Box 0)
                    box = 1
                    try:
                        if hasattr(runner, 'metadata') and runner.metadata and 'TRAP' in runner.metadata:
                            box = int(runner.metadata['TRAP'])
                        else:
                            # Try prefix "1. Dog Name"
                            name_match = re.match(r'^(\d+)\.', runner.runner_name)
                            if name_match:
                                box = int(name_match.group(1))
                    except: pass
                        
                    # Dog
                    c.execute("SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName = ?", (d_name,))
                    g_row = c.fetchone()
                    if g_row:
                        gid = g_row[0]
                    else:
                        c.execute("INSERT INTO Greyhounds (GreyhoundName) VALUES (?)", (d_name,))
                        gid = c.lastrowid
                        
                    # Entry
                    c.execute("SELECT EntryID FROM GreyhoundEntries WHERE RaceID = ? AND GreyhoundID = ?", (race_id, gid))
                    if not c.fetchone():
                        c.execute("INSERT INTO GreyhoundEntries (RaceID, GreyhoundID, Box) VALUES (?, ?, ?)", 
                                  (race_id, gid, box))

            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[ERROR] DB Auto-Create failed: {e}")
            return False

    def _capture_prices(self, market, column_name: str, interval_mins: int):
        """Capture current back and lay prices for a market"""
        market_id = market.market_id
        
        # Ensure exists in DB first
        self._ensure_race_in_db(market)
        
        # Get current back AND lay prices
        prices = self.fetcher.get_market_prices(market_id)
        if not prices:
            return
            
        # Parse market name for track/race info
        track_name, race_number = self._parse_market_name(market)
        if not track_name or not race_number:
            return
            
        # Determine if this is a PLACE market
        is_place = column_name.startswith('Place')
        
        # Derive column names for back and lay
        # For WIN: Price60Min -> Lay60Min
        # For PLACE: PlacePrice60Min -> PlaceLay60Min
        if is_place:
            back_column = column_name  # e.g., PlacePrice60Min
            # PlacePrice60Min -> PlaceLay60Min
            lay_column = column_name.replace('PlacePrice', 'PlaceLay')
        else:
            back_column = column_name  # e.g., Price60Min
            # Price60Min -> Lay60Min
            lay_column = column_name.replace('Price', 'Lay')
            
        # Get race date from market start time
        race_date = market.market_start_time.strftime('%Y-%m-%d')
        
        # Update database for each runner
        conn = sqlite3.connect(DB_PATH)
        updated_back = 0
        updated_lay = 0
        
        try:
            for selection_id, price_data in prices.items():
                back_price = price_data.get('back')
                lay_price = price_data.get('lay')
                
                # Check if already captured
                key = (market_id, selection_id, interval_mins)
                if key in self.captured:
                    continue
                    
                # Get dog name from market catalogue
                dog_name = self._get_runner_name(market, selection_id)
                if not dog_name:
                    continue
                    
                # Update BACK price
                if back_price and back_price > 0:
                    cursor = conn.execute("""
                        UPDATE GreyhoundEntries 
                        SET {column} = ?
                        WHERE RaceID IN (
                            SELECT r.RaceID FROM Races r
                            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                            JOIN Tracks t ON rm.TrackID = t.TrackID
                            WHERE t.TrackName LIKE ? 
                            AND rm.MeetingDate = ?
                            AND r.RaceNumber = ?
                        )
                        AND GreyhoundID IN (
                            SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName LIKE ?
                        )
                    """.format(column=back_column), (back_price, f"%{track_name}%", race_date, race_number, f"%{dog_name}%"))
                    
                    if cursor.rowcount > 0:
                        updated_back += 1
                
                # Update LAY price
                if lay_price and lay_price > 0:
                    cursor = conn.execute("""
                        UPDATE GreyhoundEntries 
                        SET {column} = ?
                        WHERE RaceID IN (
                            SELECT r.RaceID FROM Races r
                            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                            JOIN Tracks t ON rm.TrackID = t.TrackID
                            WHERE t.TrackName LIKE ? 
                            AND rm.MeetingDate = ?
                            AND r.RaceNumber = ?
                        )
                        AND GreyhoundID IN (
                            SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName LIKE ?
                        )
                    """.format(column=lay_column), (lay_price, f"%{track_name}%", race_date, race_number, f"%{dog_name}%"))
                    
                    if cursor.rowcount > 0:
                        updated_lay += 1
                
                # Mark as captured once we've attempted both
                if back_price or lay_price:
                    self.captured[key] = True
                    
            conn.commit()
            
            if updated_back > 0 or updated_lay > 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {track_name} R{race_number}: "
                      f"Captured {updated_back} back + {updated_lay} lay for {back_column}")
                      
        except Exception as e:
            print(f"[ERROR] DB update failed: {e}")
        finally:
            conn.close()
            
    def _parse_market_name(self, market) -> Tuple[Optional[str], Optional[int]]:
        """Extract track name and race number from market name"""
        # Market name format: "Track Name R1 HH:MM" or similar
        try:
            name = market.market_name or ""
            
            # Find race number (R1, R2, etc.)
            import re
            race_match = re.search(r'R(\d+)', name, re.IGNORECASE)
            if not race_match:
                return None, None
            race_number = int(race_match.group(1))
            
            # Track name is the venue from event
            if hasattr(market, 'event') and market.event:
                track_name = market.event.venue
            else:
                # Fallback: extract from market name
                track_name = name.split(' R')[0].strip()
                
            # CLEAN TRACK NAME: Remove state suffixes e.g. "Albion Park (QLD)" -> "Albion Park"
            if track_name:
                import re
                track_name = re.sub(r'\s*\(.*?\)', '', track_name).strip()
                
            return track_name, race_number
            
        except Exception:
            return None, None
            
    def _get_runner_name(self, market, selection_id: int) -> Optional[str]:
        """Get runner name from selection ID"""
        if not hasattr(market, 'runners') or not market.runners:
            return None
            
        for runner in market.runners:
            if runner.selection_id == selection_id:
                name = runner.runner_name
                # Clean name (remove trainer suffix like "(T. Smith)")
                if '(' in name:
                    name = name.split('(')[0].strip()
                
                # Clean number prefix (e.g. "1. Dog Name" -> "Dog Name")
                import re
                name = re.sub(r'^\d+\.\s*', '', name).strip()
                    
                return name
        return None


def main():
    scraper = LivePriceScraper()
    scraper.start()


if __name__ == "__main__":
    main()
