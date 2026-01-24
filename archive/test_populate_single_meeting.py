"""
Test script to import one meeting and display the PIR (Position In Run) data
Shows API data vs database data to verify PIR codes are being captured correctly
"""

from topaz_api import TopazAPI
from greyhound_database import GreyhoundDatabase
from config import TOPAZ_API_KEY
from datetime import datetime, timedelta

def test_single_meeting():
    """Test importing a single recent meeting and display PIR data"""
    api = TopazAPI(TOPAZ_API_KEY)
    db = GreyhoundDatabase()

    print("=" * 80)
    print("TEST: IMPORT ONE MEETING WITH PIR DATA")
    print("=" * 80)

    # Use specific date for MCCATHERY test
    test_date = "2025-11-26"

    print(f"\nStep 1: Finding meetings on {test_date}...")

    # Try VIC first, then NSW
    meetings = None
    state = 'VIC'
    try:
        meetings = api.get_meetings(test_date, owning_authority_code='VIC')
    except:
        pass

    if not meetings:
        print("No VIC meetings, trying NSW...")
        state = 'NSW'
        try:
            meetings = api.get_meetings(test_date, owning_authority_code='NSW')
        except:
            pass

    if not meetings:
        print("No meetings found!")
        return

    # Pick first meeting
    meeting = meetings[0]
    track_code = meeting['trackCode']
    track_name = meeting['trackName']

    print(f"\nSelected: {track_name} ({track_code}) - {test_date}")

    # Get bulk data
    print(f"\n" + "=" * 80)
    print(f"Step 2: Getting bulk API data...")
    print("=" * 80)

    year, month, day = test_date.split('-')
    bulk_runs = api.get_bulk_runs_by_day(state, int(year), int(month), int(day))

    # Filter to this track
    track_runs = [r for r in bulk_runs if r.get('trackCode') == track_code]
    print(f"\nFound {len(track_runs)} runs at {track_name}")

    # Group by race
    races = {}
    for run in track_runs:
        race_num = run.get('raceNumber')
        if race_num not in races:
            races[race_num] = []
        races[race_num].append(run)

    print(f"Total races: {len(races)}")

    # Display Race 1 data from API
    if 1 not in races:
        print("\nRace 1 not found!")
        return

    race_1 = sorted(races[1], key=lambda x: x.get('boxNumber') or 99)

    print(f"\n" + "=" * 80)
    print("Step 3: API Data for Race 1")
    print("=" * 80)
    print(f"\n{'Box':<4} {'Greyhound':<20} {'Pos':<4} {'Time':<7} {'Margin':<10} {'PIR':<6} {'Split':<6}")
    print("-" * 80)

    for run in race_1:
        box = run.get('boxNumber') or '-'
        dog = (run.get('dogName') or '')[:20]

        if run.get('scratched'):
            pos = 'SCR'
        elif run.get('unplaced'):
            pos = 'DNF'
        else:
            pos = str(run.get('place') or '-')

        time = run.get('resultTime')
        if time is not None:
            time = f"{time:.2f}"
        else:
            time = '-'

        # Show margin in lengths
        margin = run.get('resultMarginLengths') or '-'

        pir = run.get('pir') or '-'  # PIR field!
        split = run.get('firstSplitTime')
        if split is not None:
            split = f"{split:.2f}"
        else:
            split = '-'

        print(f"{box:<4} {dog:<20} {pos:<4} {time:<7} {margin:<10} {pir:<6} {split:<6}")

    # Import Race 1 using the database's import_results_data method
    print(f"\n" + "=" * 80)
    print("Step 4: Importing Race 1 to Database...")
    print("=" * 80)

    # Convert to database format
    results = []
    for run in race_1:
        # Skip scratched dogs
        if run.get('scratched'):
            continue

        # Handle position
        if run.get('unplaced'):
            position = 'DNF'
        else:
            position = run.get('place')

        # Use margin in lengths (e.g., "10.50L") not time difference
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
            'in_run': run.get('pir', ''),  # PIR!
            'split': run.get('firstSplitTime'),
            'sire': run.get('sireName', ''),
            'dam': run.get('damName', '')
        }
        results.append(result_entry)

    distance = race_1[0].get('distanceInMetres', 0)
    grade = race_1[0].get('raceType', 'Unknown')

    results_data = {
        'race_number': 1,
        'race_name': '',
        'grade': grade,
        'distance': distance,
        'race_time': '',
        'prize_money': '',
        'results': results
    }

    # Import using the database's import method
    success = db.import_results_data(results_data, test_date, track_name)

    if success:
        print("[OK] Race 1 imported")
    else:
        print("[ERROR] Failed to import")
        return

    # Query back from database
    print(f"\n" + "=" * 80)
    print("Step 5: Database Query Results")
    print("=" * 80)

    conn = db.get_connection()
    cursor = conn.cursor()

    # Get the race_id for the race we just imported
    cursor.execute("""
        SELECT r.RaceID
        FROM Races r
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
          AND t.TrackName = ?
          AND r.RaceNumber = 1
    """, (test_date, track_name))

    race_row = cursor.fetchone()
    if not race_row:
        print("[ERROR] Could not find imported race")
        return

    race_id = race_row[0]

    cursor.execute("""
        SELECT
            g.GreyhoundName,
            ge.Box,
            ge.Position,
            ge.FinishTime,
            ge.Margin,
            ge.InRun,
            ge.Split
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE ge.RaceID = ?
        ORDER BY ge.Box
    """, (race_id,))

    results = cursor.fetchall()
    conn.close()

    print(f"\n{'Box':<4} {'Greyhound':<20} {'Pos':<4} {'Time':<7} {'Margin':<10} {'InRun':<6} {'Split':<6}")
    print("-" * 80)

    for row in results:
        greyhound = row[0][:20]
        box = row[1] or '-'
        pos = row[2] or '-'
        time = row[3]
        if time:
            time = f"{time:.2f}"
        else:
            time = '-'
        margin = row[4] or '-'
        in_run = row[5] or '-'
        split = row[6]
        if split:
            split = f"{split:.2f}"
        else:
            split = '-'

        print(f"{box:<4} {greyhound:<20} {pos:<4} {time:<7} {margin:<10} {in_run:<6} {split:<6}")

    print(f"\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nCompare the 'PIR' column from API with 'InRun' from database.")
    print("They should match (e.g., 221, 557, 332, etc.)")
    print("\nIf they match, the fix is working correctly!")


if __name__ == "__main__":
    test_single_meeting()
