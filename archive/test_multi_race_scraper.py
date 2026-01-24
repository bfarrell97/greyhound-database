"""
Test script for multi-race scraping functionality
"""

from greyhound_scraper_v2 import GreyhoundScraper
from greyhound_database import GreyhoundDatabase

def test_ballarat_meeting():
    """Test scraping all races from Ballarat meeting"""

    # Ballarat meeting with 12 races
    meeting_url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
    race_date = "2025-11-29"
    track_name = "Ballarat"

    print("=" * 80)
    print("TESTING MULTI-RACE SCRAPER")
    print("=" * 80)
    print(f"\nURL: {meeting_url}")
    print(f"Date: {race_date}")
    print(f"Track: {track_name}")
    print("\n" + "=" * 80)

    # Initialize scraper
    scraper = GreyhoundScraper(headless=False)

    try:
        # Scrape all races from the meeting
        print("\nScraping all races from the meeting...")
        all_races = scraper.scrape_all_meeting_results(meeting_url)

        if not all_races:
            print("\nERROR: No races scraped!")
            return

        print(f"\n{'=' * 80}")
        print(f"SUCCESS: Scraped {len(all_races)} races")
        print("=" * 80)

        # Display summary of each race
        for i, race_data in enumerate(all_races, 1):
            race_num = race_data.get('race_number', i)
            race_name = race_data.get('race_name', 'Unknown')
            distance = race_data.get('distance', 'Unknown')
            grade = race_data.get('grade', 'Unknown')
            num_results = len(race_data.get('results', []))

            print(f"\nRace {race_num}: {race_name}")
            print(f"  Distance: {distance}m")
            print(f"  Grade: {grade}")
            print(f"  Results: {num_results} greyhounds")

        # Save to database
        print("\n" + "=" * 80)
        print("Saving to database...")
        print("=" * 80)

        db = GreyhoundDatabase('greyhound_racing.db')

        success_count = 0
        fail_count = 0

        for i, race_data in enumerate(all_races, 1):
            race_num = race_data.get('race_number', i)
            print(f"\n[{i}/{len(all_races)}] Saving Race {race_num}...")

            try:
                success = db.import_results_data(race_data, race_date, track_name)
                if success:
                    print(f"  SUCCESS")
                    success_count += 1
                else:
                    print(f"  FAILED")
                    fail_count += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                fail_count += 1

        db.close()

        print("\n" + "=" * 80)
        print("DATABASE SAVE COMPLETE")
        print("=" * 80)
        print(f"Success: {success_count}/{len(all_races)}")
        print(f"Failed: {fail_count}/{len(all_races)}")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()

if __name__ == "__main__":
    test_ballarat_meeting()
