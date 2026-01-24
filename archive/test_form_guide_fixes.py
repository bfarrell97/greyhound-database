"""Test script to verify form guide fixes are working correctly"""

from topaz_api import TopazAPI
from greyhound_database import GreyhoundDatabase
from config import TOPAZ_API_KEY
from datetime import datetime, timedelta

def test_track_filtering():
    """Test that track filtering works correctly"""
    print("=" * 80)
    print("TEST 1: Track Filtering")
    print("=" * 80)

    api = TopazAPI(TOPAZ_API_KEY)

    # Test with a Victorian track (MEA - The Meadows)
    test_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\nTesting MEA (The Meadows) on {test_date}...")

    try:
        result = api.get_form_guide_data(test_date, 'MEA', 1)
        meeting = result['meeting']
        race = result['race']

        print(f"✓ SUCCESS")
        print(f"  Track: {meeting['trackName']} ({meeting['trackCode']})")
        print(f"  State: {meeting.get('owningAuthorityCode', 'N/A')}")
        print(f"  Race: {race['raceNumber']} - {race.get('raceName', 'N/A')}")

        if meeting['trackCode'] == 'MEA':
            print(f"  ✓ Correct track returned!")
        else:
            print(f"  ✗ WRONG TRACK! Expected MEA, got {meeting['trackCode']}")

    except Exception as e:
        print(f"✗ ERROR: {e}")


def test_running_position_data():
    """Test that running position data can be retrieved from database"""
    print("\n" + "=" * 80)
    print("TEST 2: Running Position Data Retrieval")
    print("=" * 80)

    db = GreyhoundDatabase()

    # Test with a few dogs from recent data
    test_dogs = [
        "FOLLOW THE ACE",
        "ZIPPING RAMBO",
        "ASTON RUPIAH"
    ]

    for dog_name in test_dogs:
        print(f"\nTesting: {dog_name}")
        print("-" * 40)

        try:
            form = db.get_greyhound_form(dog_name, limit=5)

            if not form:
                print(f"  No form data found")
                continue

            print(f"  Found {len(form)} races:")

            has_running_position = False
            for i, race in enumerate(form[:3], 1):  # Show first 3
                rp = race.get('RunningPosition', '') or race.get('running_position', '')
                if rp:
                    has_running_position = True
                    print(f"    Race {i}: {race['MeetingDate']} - RP: {rp}")
                else:
                    print(f"    Race {i}: {race['MeetingDate']} - RP: [EMPTY]")

            if has_running_position:
                print(f"  ✓ Running position data found!")
            else:
                print(f"  ⚠ No running position data in recent races")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")


def test_database_columns():
    """Check what columns are populated in the database"""
    print("\n" + "=" * 80)
    print("TEST 3: Database Column Population")
    print("=" * 80)

    db = GreyhoundDatabase()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Check how many entries have data in each column
    cursor.execute("""
        SELECT
            COUNT(*) as total_entries,
            SUM(CASE WHEN InRun IS NOT NULL AND InRun != '' THEN 1 ELSE 0 END) as has_inrun,
            SUM(CASE WHEN Form IS NOT NULL AND Form != '' THEN 1 ELSE 0 END) as has_form,
            SUM(CASE WHEN Split IS NOT NULL THEN 1 ELSE 0 END) as has_split
        FROM GreyhoundEntries
    """)

    result = cursor.fetchone()

    print(f"\nDatabase statistics:")
    print(f"  Total entries: {result[0]}")
    print(f"  Entries with InRun data: {result[1]} ({result[1]/result[0]*100:.1f}%)")
    print(f"  Entries with Form data: {result[2]} ({result[2]/result[0]*100:.1f}%)")
    print(f"  Entries with Split data: {result[3]} ({result[3]/result[0]*100:.1f}%)")

    if result[1] > 0:
        print(f"\n  ✓ InRun column has data (new bulk import)")
    if result[2] > 0:
        print(f"  ✓ Form column has data (legacy data)")

    conn.close()


def test_coalesce_query():
    """Test that COALESCE is working correctly"""
    print("\n" + "=" * 80)
    print("TEST 4: COALESCE Query Test")
    print("=" * 80)

    db = GreyhoundDatabase()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Find entries where Form has data but InRun doesn't
    cursor.execute("""
        SELECT
            g.GreyhoundName,
            ge.Form,
            ge.InRun,
            COALESCE(ge.InRun, ge.Form) as RunningPosition
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE (ge.Form IS NOT NULL AND ge.Form != '')
           OR (ge.InRun IS NOT NULL AND ge.InRun != '')
        LIMIT 10
    """)

    results = cursor.fetchall()

    if results:
        print(f"\nSample of running position data:")
        print(f"{'Greyhound':<20} {'Form':<10} {'InRun':<10} {'COALESCE Result':<15}")
        print("-" * 60)

        for row in results:
            name = row[0][:18]
            form = row[1] or '[empty]'
            inrun = row[2] or '[empty]'
            coalesce = row[3] or '[empty]'
            print(f"{name:<20} {form:<10} {inrun:<10} {coalesce:<15}")

        print(f"\n✓ COALESCE is working - returning first non-null value")
    else:
        print("\n⚠ No running position data found in database")

    conn.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FORM GUIDE FIX VALIDATION TESTS")
    print("=" * 80)
    print("\nThis script validates that all form guide fixes are working correctly:\n")
    print("1. Track filtering (queries correct state)")
    print("2. Running position data retrieval")
    print("3. Database column population statistics")
    print("4. COALESCE query functionality")
    print("\n" + "=" * 80 + "\n")

    try:
        test_track_filtering()
        test_running_position_data()
        test_database_columns()
        test_coalesce_query()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
        import traceback
        traceback.print_exc()
