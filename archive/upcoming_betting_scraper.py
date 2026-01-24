"""
Upcoming Betting Races Scraper
Fetches upcoming greyhound races with current odds for ML predictions
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
import requests
import json


class UpcomingBettingScraper:
    """Scraper for upcoming betting races with current odds"""

    def __init__(self, db_path: str = "greyhound_racing.db"):
        self.db_path = db_path
        self.topaz_api = TopazAPI(TOPAZ_API_KEY)

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def create_upcoming_betting_tables(self):
        """Create tables for storing upcoming betting races"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS UpcomingBettingRaces (
                UpcomingBettingRaceID INTEGER PRIMARY KEY AUTOINCREMENT,
                MeetingDate TEXT NOT NULL,
                TrackCode TEXT NOT NULL,
                TrackName TEXT NOT NULL,
                RaceNumber INTEGER NOT NULL,
                RaceTime TEXT,
                Distance INTEGER,
                RaceType TEXT,
                LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(MeetingDate, TrackCode, RaceNumber)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS UpcomingBettingRunners (
                UpcomingBettingRunnerID INTEGER PRIMARY KEY AUTOINCREMENT,
                UpcomingBettingRaceID INTEGER NOT NULL,
                GreyhoundName TEXT NOT NULL,
                BoxNumber INTEGER,
                CurrentOdds REAL,
                TrainerName TEXT,
                Form TEXT,
                BestTime TEXT,
                Weight REAL,
                LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (UpcomingBettingRaceID) REFERENCES UpcomingBettingRaces(UpcomingBettingRaceID)
            )
        """)

        conn.commit()
        conn.close()
        print("[OK] Created upcoming betting races tables")

    def fetch_upcoming_betting_races(self, date: str) -> List[Dict[str, Any]]:
        """
        Fetch upcoming betting races for a specific date from Topaz API

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of race dictionaries with runner information
        """
        print(f"\n[Fetching] Upcoming betting races for {date}...")

        all_races = []

        # Get meetings from Australian mainland states only (excluding TAS and NZ)
        # TAS and NZ are excluded as the model filters them out anyway
        states = ['NSW', 'VIC', 'QLD', 'SA', 'WA']
        all_meetings = []

        for state in states:
            try:
                state_meetings = self.topaz_api.get_meetings(date, owning_authority_code=state)
                all_meetings.extend(state_meetings)
                if len(state_meetings) > 0:
                    print(f"  [OK] Found {len(state_meetings)} meetings in {state}")
            except Exception as e:
                print(f"  [WARNING] Failed to get {state} meetings: {e}")
                continue

        meetings = all_meetings
        print(f"[OK] Found {len(meetings)} total meetings across all states")

        # Track name mapping: Topaz API name -> Betfair name
        track_name_mapping = {
            'Sandown (SAP)': 'Sandown Park',
        }

        for meeting in meetings:
            track_code = meeting['trackCode']
            track_name = meeting['trackName']
            # Apply Betfair track name mapping
            betfair_track_name = track_name_mapping.get(track_name, track_name)
            meeting_date = meeting['meetingDate'].split('T')[0]

            print(f"\n  [Processing] {betfair_track_name} ({track_code})")

            # Get all races for this meeting
            try:
                races = self.topaz_api.get_races(meeting['meetingId'])
                print(f"    Found {len(races)} races")

                for race in races:
                    race_info = {
                        'meeting_date': meeting_date,
                        'track_code': track_code,
                        'track_name': betfair_track_name,  # Use Betfair-compatible name
                        'race_number': race['raceNumber'],
                        'race_time': race.get('startTime', race.get('raceStart')),
                        'distance': race.get('distance'),
                        'race_type': race.get('raceType'),
                        'runners': []
                    }

                    # Get runner information
                    for run in race.get('runs', []):
                        if not run.get('scratched', False):  # Skip scratched runners
                            box_number = run.get('boxNumber')

                            # Only include runners with box numbers (boxes drawn)
                            # Skip races where boxes haven't been drawn yet
                            if box_number is not None:
                                runner = {
                                    'greyhound_name': run.get('dogName'),
                                    'box_number': box_number,
                                    'trainer_name': run.get('trainerName'),
                                    'form': run.get('last5'),
                                    'best_time': run.get('bestTime'),
                                    'weight': run.get('weightInKg'),
                                    'current_odds': None  # Will be populated from betting API
                                }
                                race_info['runners'].append(runner)

                    all_races.append(race_info)

            except Exception as e:
                print(f"    [ERROR] Failed to get races for {track_name}: {e}")
                continue

        print(f"\n[OK] Fetched {len(all_races)} upcoming betting races total")
        return all_races

    def fetch_betfair_odds(self, race_info: Dict[str, Any]) -> Dict[int, float]:
        """
        Fetch current odds from Betfair for a specific race

        Args:
            race_info: Race information dictionary

        Returns:
            Dictionary mapping box number to current odds
        """
        try:
            from betfair_odds_fetcher import BetfairOddsFetcher

            # Create Betfair odds fetcher if not already created
            if not hasattr(self, 'betfair_fetcher'):
                self.betfair_fetcher = BetfairOddsFetcher()
                try:
                    self.betfair_fetcher.login()
                    print("    [OK] Logged in to Betfair")
                except Exception as e:
                    print(f"    [WARNING] Betfair login failed: {e}")
                    return {}

            # Get odds for this race
            odds = self.betfair_fetcher.get_race_odds_by_box(
                race_info['track_name'],
                race_info['race_number'],
                race_info.get('race_time')
            )
            return odds

        except ImportError as e:
            print(f"    [WARNING] Failed to import Betfair odds fetcher: {e}")
            return {}
        except Exception as e:
            print(f"    [WARNING] Failed to fetch Betfair odds: {e}")
            return {}

    def save_upcoming_betting_races_to_db(self, races: List[Dict[str, Any]]):
        """
        Save upcoming betting races to database

        Args:
            races: List of race dictionaries from fetch_upcoming_betting_races()
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        races_saved = 0
        runners_saved = 0

        for race in races:
            try:
                # Insert or update race
                cursor.execute("""
                    INSERT INTO UpcomingBettingRaces
                    (MeetingDate, TrackCode, TrackName, RaceNumber, RaceTime, Distance, RaceType, LastUpdated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(MeetingDate, TrackCode, RaceNumber)
                    DO UPDATE SET
                        RaceTime=excluded.RaceTime,
                        Distance=excluded.Distance,
                        RaceType=excluded.RaceType,
                        LastUpdated=CURRENT_TIMESTAMP
                """, (
                    race['meeting_date'],
                    race['track_code'],
                    race['track_name'],
                    race['race_number'],
                    race['race_time'],
                    race['distance'],
                    race['race_type']
                ))

                upcoming_betting_race_id = cursor.lastrowid
                if upcoming_betting_race_id == 0:
                    # Race already exists, get its ID
                    cursor.execute("""
                        SELECT UpcomingBettingRaceID FROM UpcomingBettingRaces
                        WHERE MeetingDate=? AND TrackCode=? AND RaceNumber=?
                    """, (race['meeting_date'], race['track_code'], race['race_number']))
                    upcoming_betting_race_id = cursor.fetchone()[0]

                races_saved += 1

                # Fetch current odds from Betfair (if implemented)
                odds_map = self.fetch_betfair_odds(race)
                if odds_map:
                    print(f"    [OK] Got odds for {len(odds_map)} boxes")
                else:
                    print(f"    [WARNING] No odds returned for {race['track_name']} R{race['race_number']}")

                # Only save races that have boxes drawn
                if len(race['runners']) == 0:
                    continue

                # Delete old runners for this race to prevent duplicates (do this AFTER checking we have new data)
                cursor.execute("""
                    DELETE FROM UpcomingBettingRunners
                    WHERE UpcomingBettingRaceID = ?
                """, (upcoming_betting_race_id,))

                # Insert new runners
                for runner in race['runners']:
                    box_number = runner['box_number']
                    current_odds = odds_map.get(box_number)  # Will be None if Betfair not implemented

                    cursor.execute("""
                        INSERT INTO UpcomingBettingRunners
                        (UpcomingBettingRaceID, GreyhoundName, BoxNumber, CurrentOdds,
                         TrainerName, Form, BestTime, Weight, LastUpdated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        upcoming_betting_race_id,
                        runner['greyhound_name'],
                        runner['box_number'],
                        current_odds,
                        runner['trainer_name'],
                        runner['form'],
                        runner['best_time'],
                        runner['weight']
                    ))
                    runners_saved += 1

            except Exception as e:
                print(f"[ERROR] Failed to save race {race['track_code']} R{race['race_number']}: {e}")
                continue

        conn.commit()
        conn.close()

        print(f"\n[OK] Saved {races_saved} races and {runners_saved} runners to database")

    def scrape_date(self, date: str):
        """
        Complete scraping workflow for a specific date

        Args:
            date: Date in YYYY-MM-DD format
        """
        print("=" * 80)
        print(f"UPCOMING BETTING RACES SCRAPER - {date}")
        print("=" * 80)

        # Ensure tables exist
        self.create_upcoming_betting_tables()

        # Fetch races from Topaz API
        races = self.fetch_upcoming_betting_races(date)

        if not races:
            print(f"\n[WARNING] No upcoming betting races found for {date}")
            return

        # Save to database
        self.save_upcoming_betting_races_to_db(races)

        print("\n" + "=" * 80)
        print("SCRAPING COMPLETE")
        print("=" * 80)
        print(f"\nData saved to: {self.db_path}")
        print(f"\nTables: UpcomingBettingRaces, UpcomingBettingRunners")
        print(f"\nNOTE: Current odds require Betfair API credentials in config.py")
        print(f"      See betfair_api.py for setup instructions")

    def scrape_next_n_days(self, n_days: int = 7):
        """
        Scrape upcoming betting races for the next N days

        Args:
            n_days: Number of days to scrape (default 7)
        """
        today = datetime.now()

        for i in range(n_days):
            date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
            self.scrape_date(date)
            print("\n")


def main():
    """Test the upcoming betting scraper"""
    scraper = UpcomingBettingScraper()

    # Scrape tomorrow's races
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    scraper.scrape_date(tomorrow)


if __name__ == "__main__":
    main()
