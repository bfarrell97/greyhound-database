"""Check if Topaz API provides running position data for FOLLOW THE ACE"""

from topaz_api import TopazAPI
from greyhound_database import GreyhoundDatabase
from config import TOPAZ_API_KEY
import json

api = TopazAPI(TOPAZ_API_KEY)
db = GreyhoundDatabase()

print("=" * 80)
print("CHECKING TOPAZ API DATA FOR 'FOLLOW THE ACE'")
print("=" * 80)

# First, find a recent race for FOLLOW THE ACE from the database
print("\nStep 1: Finding recent races for FOLLOW THE ACE in database...")
conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT
        rm.MeetingDate,
        t.TrackName,
        t.TrackKey,
        r.RaceNumber,
        r.Distance,
        ge.Box,
        ge.InRun,
        ge.Form,
        ge.Position
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE g.GreyhoundName = 'FOLLOW THE ACE'
    ORDER BY rm.MeetingDate DESC
    LIMIT 5
""")

races = cursor.fetchall()
conn.close()

if not races:
    print("No races found for FOLLOW THE ACE in database")
    exit()

print(f"\nFound {len(races)} recent races:")
for i, race in enumerate(races, 1):
    print(f"  {i}. {race[0]} - {race[1]} ({race[2]}) - Race {race[3]} - Box {race[5]}")
    print(f"     Position: {race[8]}, InRun: {race[6] or '[empty]'}, Form: {race[7] or '[empty]'}")

# Take the most recent race
most_recent = races[0]
race_date = most_recent[0]
track_key = most_recent[2]  # This is the TrackKey (e.g., "Angle Park_SA_AU")
race_number = most_recent[3]

# Extract track code from TrackKey (first 3 letters of track name in uppercase)
# Or try to find it from the track name
track_name = most_recent[1]
# Simple heuristic: try common track codes
track_code_map = {
    'The Meadows': 'MEA',
    'Sandown': 'SAN',
    'Geelong': 'GEL',
    'Bendigo': 'BEN',
    'Horsham': 'HOR',
    'Angle Park': 'ANG',
    'Richmond': 'RIC',
    'Wentworth Park': 'WEN',
    'The Gardens': 'TAR',
    'Cannington': 'CAN',
    'Mandurah': 'MAN',
}
track_code = track_code_map.get(track_name, track_key.split('_')[0][:3].upper())

print(f"\n" + "=" * 80)
print(f"Step 2: Querying Topaz API for this race...")
print(f"  Date: {race_date}")
print(f"  Track: {track_code}")
print(f"  Race: {race_number}")
print("=" * 80)

try:
    # Get the race data from API
    result = api.get_form_guide_data(race_date, track_code, race_number)
    meeting = result['meeting']
    race = result['race']

    print(f"\n[OK] API call successful")
    print(f"  Meeting: {meeting['trackName']} - {meeting['meetingDate']}")
    print(f"  Race: {race['raceNumber']} - {race.get('distance')}m")

    # Find FOLLOW THE ACE in the runners
    print(f"\n" + "=" * 80)
    print("Step 3: Checking runner data for FOLLOW THE ACE...")
    print("=" * 80)

    runner_found = False
    for run in race.get('runs', []):
        if run.get('dogName') == 'FOLLOW THE ACE':
            runner_found = True
            print(f"\n[OK] Found FOLLOW THE ACE in API response!")
            print(f"\nFull runner data:")
            print(json.dumps(run, indent=2))

            # Check specifically for running position fields
            print(f"\n" + "=" * 80)
            print("RUNNING POSITION FIELDS:")
            print("=" * 80)

            running_position_fields = [
                'runLineCode',
                'inRun',
                'runningPosition',
                'position',
                'firstSplitPosition',
                'secondSplitPosition',
                'form'
            ]

            for field in running_position_fields:
                value = run.get(field)
                if value:
                    print(f"  [OK] {field}: {value}")
                else:
                    print(f"  [EMPTY] {field}: [not present or empty]")

            break

    if not runner_found:
        print(f"\n[NOT FOUND] FOLLOW THE ACE not found in API response")
        print(f"\nRunners in this race:")
        for run in race.get('runs', []):
            print(f"  - {run.get('dogName')} (Box {run.get('boxNumber')})")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("BULK ENDPOINT CHECK")
print("=" * 80)
print("\nThe bulk endpoint includes different fields. Let me also check that...")

# Extract year, month, day from race_date
year, month, day = race_date.split('-')
state = api.TRACK_STATE_MAP.get(track_code, 'VIC')

print(f"Querying bulk endpoint for {state} {year}-{month}-{day}...")

try:
    bulk_runs = api.get_bulk_runs_by_day(state, int(year), int(month), int(day))

    print(f"[OK] Got {len(bulk_runs)} runs from bulk endpoint")

    # Find FOLLOW THE ACE
    for run in bulk_runs:
        if run.get('dogName') == 'FOLLOW THE ACE' and run.get('raceNumber') == race_number:
            print(f"\n[OK] Found FOLLOW THE ACE in bulk data!")
            print(f"\nBulk endpoint data:")
            print(json.dumps(run, indent=2))

            print(f"\n" + "=" * 80)
            print("RUNNING POSITION FIELDS (BULK ENDPOINT):")
            print("=" * 80)

            running_position_fields = [
                'runLineCode',
                'inRun',
                'runningPosition',
                'position',
                'firstSplitPosition',
                'secondSplitPosition',
                'form'
            ]

            for field in running_position_fields:
                value = run.get(field)
                if value:
                    print(f"  [OK] {field}: {value}")
                else:
                    print(f"  [EMPTY] {field}: [not present or empty]")

            break

except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
