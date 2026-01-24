"""Test that PIR field is now being captured correctly"""

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
from populate_historical_data_bulk import HistoricalDataPopulatorBulk

api = TopazAPI(TOPAZ_API_KEY)

print("=" * 80)
print("TESTING PIR (Position In Run) FIELD CAPTURE")
print("=" * 80)

# Get a sample race with FOLLOW THE ACE
print("\nStep 1: Getting bulk data for 2025-03-26 (FOLLOW THE ACE race)...")
runs = api.get_bulk_runs_by_day('NSW', 2025, 3, 26)

print(f"Found {len(runs)} runs")

# Find FOLLOW THE ACE
for run in runs:
    if run.get('dogName') == 'FOLLOW THE ACE' and run.get('raceNumber') == 2:
        print(f"\nFound FOLLOW THE ACE:")
        print(f"  Dog: {run['dogName']}")
        print(f"  Track: {run['trackName']}")
        print(f"  Race: {run['raceNumber']}")
        print(f"  Box: {run['boxNumber']}")
        print(f"  Position: {run['place']}")
        print(f"\n  PIR field: {run.get('pir')} <-- This is what we need!")
        print(f"  runLineCode: {run.get('runLineCode')} <-- This is just 'Normal'/'Wide'")

        print("\n" + "=" * 80)
        print("Step 2: Testing conversion to database format...")
        print("=" * 80)

        populator = HistoricalDataPopulatorBulk()

        # Get all runs for this race
        race_runs = [r for r in runs if r.get('raceNumber') == 2 and r.get('trackName') == 'Richmond']

        if race_runs:
            results_data = populator.convert_runs_to_db_format(race_runs, 2)

            print(f"\nConverted {len(results_data['results'])} runners")

            # Find FOLLOW THE ACE in converted data
            for result in results_data['results']:
                if result['greyhound_name'] == 'FOLLOW THE ACE':
                    print(f"\nFOLLOW THE ACE converted result:")
                    print(f"  Greyhound: {result['greyhound_name']}")
                    print(f"  Box: {result['box']}")
                    print(f"  Position: {result['position']}")
                    print(f"  in_run: '{result['in_run']}' <-- This should be '557'")

                    if result['in_run'] == '557':
                        print(f"\n  [SUCCESS] PIR field is being captured correctly!")
                    else:
                        print(f"\n  [ERROR] PIR field is NOT being captured. Got: '{result['in_run']}'")
                    break

        break

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nIf the test shows 'in_run: 557', the fix is working!")
print("You need to re-run the bulk import to populate existing data.")
print("\nCommand: python populate_historical_data_bulk.py")
