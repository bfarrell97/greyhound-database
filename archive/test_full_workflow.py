"""
Test Complete Workflow: Scrape → Database → Display
"""

from greyhound_scraper_v2 import GreyhoundScraper
from greyhound_database import GreyhoundDatabase
from datetime import datetime


def test_results_workflow():
    """Test scraping results and saving to database"""
    print("=" * 80)
    print("TEST: Results Scraping -> Database Workflow")
    print("=" * 80)

    # Initialize
    scraper = GreyhoundScraper(headless=False)
    db = GreyhoundDatabase('greyhound_racing.db')

    try:
        # Step 1: Scrape results
        print("\n[1] Scraping results...")
        url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
        results = scraper.scrape_results(url)

        if results:
            print(f"[OK] Scraped {len(results.get('results', []))} entries")
            print(f"     Track: Ballarat")
            print(f"     Race {results.get('race_number')}: {results.get('distance')}m {results.get('grade')}")
        else:
            print("[ERROR] No results scraped")
            return

        # Step 2: Save to database
        print("\n[2] Saving to database...")
        success = db.import_results_data(
            results,
            meeting_date='2024-11-30',  # You should extract this from the page
            track_name='Ballarat'
        )

        if success:
            print("[OK] Data saved successfully")
        else:
            print("[ERROR] Failed to save data")
            return

        # Step 3: Verify data in database
        print("\n[3] Verifying database...")
        conn = db.get_connection()
        cursor = conn.cursor()

        # Check greyhounds added
        cursor.execute("SELECT COUNT(*) FROM Greyhounds")
        greyhound_count = cursor.fetchone()[0]
        print(f"[OK] Greyhounds in database: {greyhound_count}")

        # Check entries added
        cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries")
        entry_count = cursor.fetchone()[0]
        print(f"[OK] Total entries in database: {entry_count}")

        # Show some sample data
        print("\n[4] Sample data from database:")
        cursor.execute("""
            SELECT g.GreyhoundName, e.Box, e.Position, e.FinishTime, t.TrainerName
            FROM GreyhoundEntries e
            JOIN Greyhounds g ON e.GreyhoundID = g.GreyhoundID
            JOIN Trainers t ON e.TrainerID = t.TrainerID
            ORDER BY e.EntryID DESC
            LIMIT 5
        """)

        print("\n  Recent entries:")
        for row in cursor.fetchall():
            print(f"    {row[0]:20} Box:{row[1]} Pos:{row[2]} Time:{row[3]} Trainer:{row[4]}")

        print("\n" + "=" * 80)
        print("SUCCESS - Complete workflow tested!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()
        db.close()


def test_form_guide_workflow():
    """Test scraping form guide and saving to database"""
    print("\n" + "=" * 80)
    print("TEST: Form Guide Scraping -> Database Workflow")
    print("=" * 80)

    # Initialize
    scraper = GreyhoundScraper(headless=False)
    db = GreyhoundDatabase('greyhound_racing.db')

    try:
        # Step 1: Scrape form guide
        print("\n[1] Scraping form guide...")
        url = "https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/"
        form_data = scraper.scrape_form_guide(url)

        if form_data:
            print(f"[OK] Scraped {len(form_data.get('entries', []))} entries")
            print(f"     Track: Broken Hill")
            print(f"     Race {form_data.get('race_number')}: {form_data.get('distance')}m {form_data.get('grade')}")
        else:
            print("[ERROR] No form data scraped")
            return

        # Step 2: Save to database
        print("\n[2] Saving to database...")
        success = db.import_form_guide_data(
            form_data,
            meeting_date='2024-12-01',  # Future race date
            track_name='Broken Hill'
        )

        if success:
            print("[OK] Data saved successfully")
        else:
            print("[ERROR] Failed to save data")
            return

        # Step 3: Verify data
        print("\n[3] Verifying database...")
        conn = db.get_connection()
        cursor = conn.cursor()

        # Show upcoming race entries
        print("\n[4] Upcoming race entries:")
        cursor.execute("""
            SELECT g.GreyhoundName, e.Box, t.TrainerName, g.Starts, g.Wins
            FROM GreyhoundEntries e
            JOIN Greyhounds g ON e.GreyhoundID = g.GreyhoundID
            JOIN Trainers t ON e.TrainerID = t.TrainerID
            JOIN Races r ON e.RaceID = r.RaceID
            JOIN RaceMeetings m ON r.MeetingID = m.MeetingID
            WHERE m.MeetingDate = '2024-12-01'
            ORDER BY e.Box
        """)

        print("\n  Entries:")
        for row in cursor.fetchall():
            print(f"    Box {row[1]}: {row[0]:20} Trainer:{row[2]:20} Record:{row[4]}-{row[3]}")

        print("\n" + "=" * 80)
        print("SUCCESS - Complete workflow tested!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()
        db.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPLETE WORKFLOW TEST")
    print("=" * 80)
    print("\nThis will test:")
    print("  1. Scraping results from completed race")
    print("  2. Saving to database")
    print("  3. Scraping form guide for upcoming race")
    print("  4. Saving to database")
    print("=" * 80)

    # Test both workflows
    test_results_workflow()
    test_form_guide_workflow()

    print("\n" + "=" * 80)
    print("ALL WORKFLOWS COMPLETE")
    print("=" * 80)
