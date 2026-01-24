"""
Test script to verify DNF and SCR handling
"""

from greyhound_scraper_v2 import GreyhoundScraper
from greyhound_database import GreyhoundDatabase


def test_race_with_dnf_scr():
    """Test scraping race 4 from Ballarat which has DNF and SCR"""

    url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
    race_date = "2025-11-29"
    track_name = "Ballarat"

    print("=" * 80)
    print("Testing DNF and SCR Handling")
    print("=" * 80)
    print(f"\nScraping: {url}")
    print(f"This race should have:")
    print("  - 1 dog with DNF (Did Not Finish)")
    print("  - 2 dogs with SCR (Scratched - should NOT be imported)")
    print()

    # Scrape the race
    scraper = GreyhoundScraper(headless=True)
    race_data = scraper.scrape_results(url)

    print(f"Scraped {len(race_data['results'])} dogs from the race:")
    print()

    dnf_count = 0
    scr_count = 0

    for i, dog in enumerate(race_data['results'], 1):
        pos = dog.get('position', 'N/A')
        name = dog.get('greyhound_name', 'Unknown')
        box = dog.get('box', '?')

        if pos == 'DNF':
            dnf_count += 1
            print(f"  {i}. Box {box}: {name:<25} Position: DNF ✓")
        elif pos == 'SCR':
            scr_count += 1
            print(f"  {i}. Box {box}: {name:<25} Position: SCR ❌ (SHOULD NOT APPEAR)")
        else:
            print(f"  {i}. Box {box}: {name:<25} Position: {pos}")

    print()
    print("=" * 80)
    print("Results:")
    print("=" * 80)

    if dnf_count == 1:
        print(f"✓ Correctly found {dnf_count} DNF dog")
    else:
        print(f"❌ Expected 1 DNF dog, found {dnf_count}")

    if scr_count == 0:
        print(f"✓ Correctly skipped SCR dogs (found {scr_count})")
    else:
        print(f"❌ SCR dogs should be skipped, but found {scr_count}")

    # Now import to database
    print()
    print("Importing to database...")
    db = GreyhoundDatabase('greyhound_racing.db')
    success = db.import_results_data(race_data, race_date, track_name)

    if success:
        print("✓ Successfully imported to database")

        # Check database
        conn = db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT g.GreyhoundName, ge.Position, ge.Box
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.RaceNumber = 4
            ORDER BY ge.Box
        """)

        db_entries = cursor.fetchall()
        print(f"\nDatabase contains {len(db_entries)} entries for Race 4:")
        for name, pos, box in db_entries:
            print(f"  Box {box}: {name:<25} Position: {pos}")

        # Check for DNF in database
        cursor.execute("""
            SELECT COUNT(*) FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.RaceNumber = 4 AND ge.Position = 'DNF'
        """)
        dnf_in_db = cursor.fetchone()[0]

        print()
        if dnf_in_db == 1:
            print(f"✓ Database correctly stored DNF position")
        else:
            print(f"❌ Expected 1 DNF in database, found {dnf_in_db}")
    else:
        print("❌ Failed to import to database")

    print()
    print("=" * 80)


if __name__ == '__main__':
    test_race_with_dnf_scr()
