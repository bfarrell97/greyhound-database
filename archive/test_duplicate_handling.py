"""
Test script to verify duplicate handling

Tests that re-scraping the same race overwrites existing data instead of creating duplicates
"""

from greyhound_scraper_v2 import GreyhoundScraper
from greyhound_database import GreyhoundDatabase
import sqlite3


def check_entry_count(db, race_date, track_name, race_number):
    """Check how many entries exist for a specific race"""
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
        AND t.TrackName = ?
        AND r.RaceNumber = ?
    """, (race_date, track_name, race_number))

    return cursor.fetchone()[0]


def get_race_details(db, race_date, track_name, race_number):
    """Get detailed race entry information"""
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            g.GreyhoundName,
            ge.Box,
            ge.Position,
            ge.FinishTime,
            ge.EntryID
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
        AND t.TrackName = ?
        AND r.RaceNumber = ?
        ORDER BY ge.Box
    """, (race_date, track_name, race_number))

    return cursor.fetchall()


def test_duplicate_prevention():
    """Test that scraping the same race twice updates instead of duplicating"""

    # Test with a known race URL
    url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
    race_date = "2025-11-29"
    track_name = "Ballarat"

    print("=" * 80)
    print("TESTING DUPLICATE HANDLING")
    print("=" * 80)

    db = GreyhoundDatabase('greyhound_racing.db')

    # Scrape the race for the first time
    print("\n1. Scraping race for the FIRST time...")
    scraper = GreyhoundScraper(headless=True)
    race_data = scraper.scrape_results(url)
    scraper.quit()

    success = db.import_results_data(race_data, race_date, track_name)

    if not success:
        print("❌ Failed to import race data!")
        return

    # Check entry count
    count_1 = check_entry_count(db, race_date, track_name, 1)
    print(f"\n   Entry count after first scrape: {count_1}")

    details_1 = get_race_details(db, race_date, track_name, 1)
    print(f"\n   Race details:")
    for dog, box, pos, time, entry_id in details_1:
        print(f"     Box {box}: {dog:<25} Pos {pos}  Time {time}  (EntryID: {entry_id})")

    # Scrape the SAME race again
    print("\n\n2. Scraping the SAME race for the SECOND time...")
    scraper = GreyhoundScraper(headless=True)
    race_data = scraper.scrape_results(url)
    scraper.quit()

    success = db.import_results_data(race_data, race_date, track_name)

    if not success:
        print("❌ Failed to import race data on second attempt!")
        return

    # Check entry count again
    count_2 = check_entry_count(db, race_date, track_name, 1)
    print(f"\n   Entry count after second scrape: {count_2}")

    details_2 = get_race_details(db, race_date, track_name, 1)
    print(f"\n   Race details:")
    for dog, box, pos, time, entry_id in details_2:
        print(f"     Box {box}: {dog:<25} Pos {pos}  Time {time}  (EntryID: {entry_id})")

    # Verify results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    if count_1 == count_2:
        print(f"✓ SUCCESS: Entry count remained the same ({count_1} entries)")
        print("  Duplicates were prevented - existing entries were updated!")
    else:
        print(f"❌ FAILURE: Entry count changed from {count_1} to {count_2}")
        print("  Duplicates were created instead of updating existing entries!")

    # Check if EntryIDs are the same (they should be if entries were updated)
    entry_ids_1 = set(e[4] for e in details_1)
    entry_ids_2 = set(e[4] for e in details_2)

    if entry_ids_1 == entry_ids_2:
        print(f"✓ SUCCESS: EntryIDs remained the same (entries were updated in-place)")
    else:
        print(f"⚠ WARNING: EntryIDs changed (new entries may have been created)")
        print(f"  Before: {sorted(entry_ids_1)}")
        print(f"  After:  {sorted(entry_ids_2)}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_duplicate_prevention()
