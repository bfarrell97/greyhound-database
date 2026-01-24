"""
Populate database with historical greyhound racing data from Topaz API
Uses BULK monthly endpoints for fast import with split times included
Commits after each race to ensure data integrity
"""

from datetime import datetime, timedelta
from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
from greyhound_database import GreyhoundDatabase
import time
import sys
import argparse
from collections import defaultdict

class HistoricalDataPopulatorBulk:
    def __init__(self):
        self.api = TopazAPI(TOPAZ_API_KEY)
        self.db = GreyhoundDatabase()
        self.stats = {
            'months_processed': 0,
            'runs_imported': 0,
            'races_imported': 0,
            'errors': 0
        }

    def populate_date_range(self, start_date: str, end_date: str, states=None):
        """
        Populate database with race data using bulk monthly endpoint

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            states: List of state codes to fetch (default: all Australian states + NZ)
        """
        if states is None:
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Generate list of (year, month) tuples to process
        months_to_process = []
        current = start
        while current <= end:
            year_month = (current.year, current.month)
            if year_month not in months_to_process:
                months_to_process.append(year_month)
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        total_months = len(months_to_process) * len(states)

        print(f"Populating data from {start_date} to {end_date}")
        print(f"States: {', '.join(states)}")
        print(f"Total months to process: {len(months_to_process)} months × {len(states)} states = {total_months} API calls")
        print("=" * 80)

        month_count = 0
        for year, month in months_to_process:
            month_name = datetime(year, month, 1).strftime('%B %Y')
            print(f"\n{month_name}")

            for state in states:
                month_count += 1
                print(f"  [{month_count}/{total_months}] {state}...", end=" ", flush=True)

                try:
                    self.populate_month(state, year, month)
                    print(f"OK ({self.stats['runs_imported']} total runs)")
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user!")
                    self.print_summary()
                    sys.exit(0)
                except Exception as e:
                    print(f"ERROR: {e}")
                    self.stats['errors'] += 1

        self.print_summary()

    def populate_month(self, state: str, year: int, month: int):
        """Populate database with all runs for a specific state and month - commit after each race"""
        # Get bulk runs for this month
        runs = self.api.get_bulk_runs_by_month(state, year, month)

        if not runs:
            return

        # Group runs by meeting and race
        meetings = defaultdict(lambda: defaultdict(list))

        for run in runs:
            # Skip runs without results (future races or scratched before race day)
            if run.get('resultTime') is None and not run.get('scratched'):
                continue

            meeting_date = run['meetingDate'][:10]  # Extract YYYY-MM-DD
            track_name = run['trackName']
            race_number = run['raceNumber']

            meeting_key = (meeting_date, track_name)
            meetings[meeting_key][race_number].append(run)

        # Import races with commit after each race
        import io
        races_processed = 0
        errors = 0

        # Suppress output for all imports
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            for (meeting_date, track_name), races in meetings.items():
                for race_number, race_runs in races.items():
                    try:
                        # Convert to database format
                        results_data = self.convert_runs_to_db_format(race_runs, race_number)

                        # Import the race and commit immediately
                        success = self.db.import_results_data(results_data, meeting_date, track_name)

                        if success:
                            # Commit after each race
                            conn = self.db.get_connection()
                            conn.commit()

                            races_processed += 1
                            self.stats['runs_imported'] += len(race_runs)

                    except Exception as e:
                        errors += 1

            self.stats['races_imported'] += races_processed
            self.stats['errors'] += errors

        finally:
            sys.stdout = old_stdout

        self.stats['months_processed'] += 1

    def convert_runs_to_db_format(self, runs, race_number):
        """
        Convert Topaz API bulk runs to database import format

        Args:
            runs: List of run dicts from bulk endpoint
            race_number: Race number

        Returns:
            Dictionary in format expected by import_results_data()
        """
        results = []

        # Get race-level info from first run
        first_run = runs[0]
        distance = first_run.get('distanceInMetres')
        grade = first_run.get('raceType', '')
        race_time = first_run.get('meetingDate', '')

        for run in runs:
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
                'starting_price': str(run.get('startPrice', '')),
                'weight': run.get('weightInKg'),
                'in_run': run.get('pir', ''),  # PIR = Position In Run (e.g., "557" = pos 5, 5, 7 at sectionals)
                'split': run.get('firstSplitTime'),  # First split time is available!
                'sire': run.get('sireName', ''),
                'dam': run.get('damName', '')
            }

            results.append(result_entry)

        return {
            'race_number': race_number,
            'race_name': '',
            'grade': grade,
            'distance': distance,
            'race_time': race_time,
            'prize_money': '',
            'results': results
        }

    def print_summary(self):
        """Print summary of data population"""
        print("\n" + "=" * 80)
        print("POPULATION SUMMARY")
        print("=" * 80)
        print(f"Months processed: {self.stats['months_processed']}")
        print(f"Races imported: {self.stats['races_imported']}")
        print(f"Runs imported: {self.stats['runs_imported']}")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 80)


def main():
    """Main function with interactive prompts"""
    print("=" * 80)
    print("GREYHOUND HISTORICAL DATA POPULATION (BULK - OPTIMIZED)")
    print("=" * 80)
    print("Using BULK monthly endpoint with batch commits for maximum speed!")
    print("=" * 80)

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Ask for start date
    print(f"\nStart Date (YYYY-MM-DD)")
    print("Press Enter for default (90 days ago)")
    start_input = input("> ").strip()

    if start_input:
        try:
            datetime.strptime(start_input, '%Y-%m-%d')
            start_date = start_input
        except ValueError:
            print("Invalid date format! Using default (90 days ago)")
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    # Ask for end date
    print(f"\nEnd Date (YYYY-MM-DD)")
    print(f"Press Enter for default (yesterday: {yesterday})")
    end_input = input("> ").strip()

    if end_input:
        try:
            datetime.strptime(end_input, '%Y-%m-%d')
            end_date = end_input
        except ValueError:
            print(f"Invalid date format! Using yesterday ({yesterday})")
            end_date = yesterday
    else:
        end_date = yesterday

    # Ask for states
    print("\nStates to import:")
    print("1. All states (VIC, NSW, QLD, SA, WA, TAS, NZ)")
    print("2. VIC only")
    print("3. NSW only")
    print("4. QLD only")
    print("5. Custom selection")

    choice = input("\nSelect option (1-5) [default: 1]: ").strip()

    if choice == '2':
        states = ['VIC']
    elif choice == '3':
        states = ['NSW']
    elif choice == '4':
        states = ['QLD']
    elif choice == '5':
        print("\nEnter state codes separated by spaces (e.g., VIC NSW QLD)")
        print("Available: VIC, NSW, QLD, SA, WA, TAS, NZ")
        state_input = input("> ").strip().upper()
        states = [s for s in state_input.split() if s in ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']]
        if not states:
            print("No valid states! Using all states.")
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']
    else:
        states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']

    # Show summary
    print("\n" + "=" * 80)
    print("IMPORT SUMMARY")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"States: {', '.join(states)}")

    # Calculate months
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    months = []
    current = start
    while current <= end:
        if (current.year, current.month) not in months:
            months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    total_api_calls = len(months) * len(states)

    print(f"Months to process: {len(months)}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Estimated time: {total_api_calls * 1}s - {total_api_calls * 3}s")
    print("=" * 80)

    # Confirmation
    response = input("\nStart import? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return

    populator = HistoricalDataPopulatorBulk()

    print("\nStarting import...")
    print("=" * 80)
    start_time = time.time()

    try:
        populator.populate_date_range(start_date, end_date, states)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    if total_api_calls > 0:
        print(f"Average: {elapsed/total_api_calls:.1f}s per API call")


if __name__ == "__main__":
    main()
