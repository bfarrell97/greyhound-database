"""
Explore what data the Topaz API provides that we might not be capturing.
"""
import json
from topaz_api import TopazAPI
from config import TOPAZ_API_KEY

api = TopazAPI(TOPAZ_API_KEY)

# Get a sample of bulk runs data
print("=" * 80)
print("EXPLORING TOPAZ API DATA FIELDS")
print("=" * 80)

# 1. Get bulk runs for a recent day
print("\n1. BULK RUNS DATA (per run/entry)")
print("-" * 40)
try:
    runs = api.get_bulk_runs_by_day("VIC", 2025, 1, 1)
    if runs:
        sample = runs[0]
        print(f"Sample run has {len(sample.keys())} fields:")
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")

# 2. Get meeting data
print("\n2. MEETING DATA")
print("-" * 40)
try:
    meetings = api.get_meetings("2025-01-01", owning_authority_code="VIC")
    if meetings:
        sample = meetings[0]
        print(f"Sample meeting has {len(sample.keys())} fields:")
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")

# 3. Get race data
print("\n3. RACE DATA (from meeting)")
print("-" * 40)
try:
    if meetings:
        meeting_id = meetings[0]['meetingId']
        races = api.get_races(meeting_id)
        if races:
            sample_race = races[0]
            print(f"Sample race has {len(sample_race.keys())} fields:")
            for key in sorted(sample_race.keys()):
                value = sample_race[key]
                if isinstance(value, (dict, list)):
                    if key == 'runs' and value:
                        print(f"  {key}: list with {len(value)} runners")
                        print(f"    -> Runner fields: {list(value[0].keys())}")
                    else:
                        print(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")

# 4. Full dump of one run for detailed inspection
print("\n4. FULL RUN DATA DUMP")
print("-" * 40)
try:
    runs = api.get_bulk_runs_by_day("VIC", 2025, 1, 1)
    if runs:
        print(json.dumps(runs[0], indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")

# 5. Check what's in the form guide/race runs
print("\n5. FORM GUIDE RUNNER DATA (from races endpoint)")
print("-" * 40)
try:
    if meetings and races:
        if races[0].get('runs'):
            runner = races[0]['runs'][0]
            print(f"Sample runner has {len(runner.keys())} fields:")
            print(json.dumps(runner, indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")
