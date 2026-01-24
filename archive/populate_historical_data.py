"""
Populate database with historical greyhound racing data from Topaz API
Fetches race results from 2020 to present
"""

from datetime import datetime, timedelta
from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
from greyhound_database import GreyhoundDatabase
import time
import sys

class HistoricalDataPopulator:
    def __init__(self):
        self.api = TopazAPI(TOPAZ_API_KEY)
        self.db = GreyhoundDatabase()
        self.stats = {
            'meetings_processed': 0,
            'races_processed': 0,
            'races_imported': 0,
            'errors': 0,
            'skipped_no_results': 0
        }

    def populate_date_range(self, start_date: str, end_date: str, states=None):
        """
        Populate database with race data for a date range

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            states: List of state codes to fetch (default: all Australian states + NZ)
        """
        if states is None:
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'NZ']

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start

        total_days = (end - start).days + 1

        print(f"Populating data from {start_date} to {end_date} ({total_days} days)")
        print(f"States: {', '.join(states)}")
        print("=" * 80)

        day_count = 0
        while current <= end:
            day_count += 1
            date_str = current.strftime('%Y-%m-%d')
            print(f"\n[Day {day_count}/{total_days}] {date_str}")

            try:
                self.populate_date(date_str, states)
            except KeyboardInterrupt:
                print("\n\nInterrupted by user!")
                self.print_summary()
                sys.exit(0)
            except Exception as e:
                print(f"  ERROR processing {date_str}: {e}")
                self.stats['errors'] += 1

            current += timedelta(days=1)

            # Small delay to avoid overwhelming the API
            time.sleep(0.2)

            # Print progress every 10 days
            if day_count % 10 == 0:
                self.print_progress()

        self.print_summary()

    def populate_date(self, date: str, states):
        """Populate database with all race data for a specific date"""
        date_meetings = 0
        date_races = 0

        for state in states:
            try:
                # Get all meetings for this state on this date
                meetings = self.api.get_meetings(date, owning_authority_code=state)

                for meeting in meetings:
                    meeting_id = meeting['meetingId']
                    track_name = meeting['trackName']
                    track_code = meeting['trackCode']

                    # Get all races for this meeting
                    races = self.api.get_races(meeting_id)

                    for race in races:
                        # Only process races that have results
                        if 'runs' not in race or not race['runs']:
                            continue

                        # Check if this race has finished (has results)
                        first_run = race['runs'][0]
                        if first_run.get('resultTime') is None and not first_run.get('scratched'):
                            # Skip races that haven't been run yet
                            self.stats['skipped_no_results'] += 1
                            continue

                        # Convert to database format
                        results_data = self.convert_race_to_db_format(race)

                        # Import the race
                        success = self.db.import_results_data(results_data, date, track_name)

                        if success:
                            date_races += 1
                        else:
                            self.stats['errors'] += 1

                    date_meetings += 1

            except Exception as e:
                # Some states might not have data for certain dates (404 errors)
                if "404" not in str(e) and "Client Error" not in str(e):
                    print(f"  ERROR fetching {state}: {e}")
                    self.stats['errors'] += 1

        self.stats['meetings_processed'] += date_meetings
        self.stats['races_imported'] += date_races

        if date_meetings > 0:
            print(f"  {date_meetings} meetings, {date_races} races imported")

    def convert_race_to_db_format(self, race):
        """
        Convert Topaz API race format to database import format

        Args:
            race: Race dict from Topaz API

        Returns:
            Dictionary in format expected by import_results_data()
        """
        results = []

        for run in race['runs']:
            # Skip scratched dogs
            if run.get('scratched'):
                continue

            # Handle position
            if run.get('unplaced'):
                position = 'DNF'
            else:
                position = run.get('place')

            # Handle margin - use lengths (e.g. "10.50L") not time difference
            margin = run.get('resultMarginLengths') or run.get('resultMargin', '')
            if isinstance(margin, (int, float)):
                margin = str(margin)

            result_entry = {
                'greyhound_name': run.get('dogName', ''),
                'box': run.get('boxNumber') or run.get('rugNumber'),
                'trainer': run.get('trainerName', ''),
                'position': position,
                'finish_time': run.get('resultTime'),
                'margin': margin,
                'starting_price': str(run.get('startingPrice') or run.get('startPrice', '')),
                'weight': run.get('weightInKg'),
                'in_run': run.get('pir', ''),  # PIR = Position In Run (e.g., "557" = pos 5, 5, 7 at sectionals)
                'split': None,  # First sectional not available in this endpoint
                'sire': run.get('sireName', ''),
                'dam': run.get('damName', '')
            }

            results.append(result_entry)

        return {
            'race_number': race['raceNumber'],
            'race_name': race.get('raceName', ''),
            'grade': race.get('raceType', ''),
            'distance': race['distance'],
            'race_time': race.get('startTime', ''),
            'prize_money': str(race.get('prizeMoney1', '')),
            'results': results
        }

    def print_progress(self):
        """Print current progress"""
        print("\n" + "-" * 80)
        print("PROGRESS UPDATE:")
        print(f"  Meetings processed: {self.stats['meetings_processed']}")
        print(f"  Races imported: {self.stats['races_imported']}")
        print(f"  Errors: {self.stats['errors']}")
        print("-" * 80)

    def print_summary(self):
        """Print summary of data population"""
        print("\n" + "=" * 80)
        print("POPULATION SUMMARY")
        print("=" * 80)
        print(f"Meetings processed: {self.stats['meetings_processed']}")
        print(f"Races imported: {self.stats['races_imported']}")
        print(f"Races skipped (no results): {self.stats['skipped_no_results']}")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 80)


def main():
    """Main function to run the population"""
    populator = HistoricalDataPopulator()

    # Import 2025 up to yesterday (today might have incomplete results)
    start_date = "2025-01-01"
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = yesterday

    print("=" * 80)
    print("GREYHOUND HISTORICAL DATA POPULATION")
    print("=" * 80)
    print(f"This will populate the database with race results from {start_date} to {end_date}")

    # Calculate days
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1

    print(f"Total days: {total_days}")
    print(f"Estimated time: 5-15 minutes")
    print("\nYou can press Ctrl+C at any time to stop the import.")
    print("Data is committed after each race, so you can resume later.")
    print()

    response = input("Do you want to continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        print("\nStarting import...")
        populator.populate_date_range(start_date, end_date)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
