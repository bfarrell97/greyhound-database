"""Test the bulk runs endpoint to check for split times"""

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
import json

api = TopazAPI(TOPAZ_API_KEY)

print("Testing bulk runs endpoint for VIC 2024-12-01...")
print("=" * 80)

try:
    runs = api.get_bulk_runs_by_day('VIC', 2024, 12, 1)
    print(f"SUCCESS! Found {len(runs)} runs")

    if runs:
        print("\nFirst run sample:")
        print(json.dumps(runs[0], indent=2))

        # Check for split time fields
        print("\n" + "=" * 80)
        print("Checking for split time fields...")
        first_run = runs[0]

        split_fields = [
            'firstSplitTime', 'firstSplitPosition',
            'secondSplitTime', 'secondSplitPosition',
            'splitTimes', 'sectionals'
        ]

        for field in split_fields:
            if field in first_run:
                print(f"[OK] Found: {field} = {first_run[field]}")

        # Check several runs to see if firstSplitTime is ever populated
        print("\n" + "=" * 80)
        print("Checking first 10 runs for non-zero firstSplitTime...")
        for i, run in enumerate(runs[:20]):
            if run.get('firstSplitTime') and run.get('firstSplitTime') != 0:
                print(f"\nRun {i}: {run.get('dogName')}")
                print(f"  firstSplitTime: {run.get('firstSplitTime')}")
                print(f"  firstSplitPosition: {run.get('firstSplitPosition')}")
                print(f"  resultTime: {run.get('resultTime')}")
                break
        else:
            print("No runs with non-zero firstSplitTime found in first 20 runs")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
