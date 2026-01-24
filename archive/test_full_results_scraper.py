"""
Test the full results scraper method
"""

from greyhound_scraper_v2 import GreyhoundScraper
import json


def test_full_results_scraper():
    """Test the complete scrape_results method"""
    print("=" * 80)
    print("Testing Full Results Scraper")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
        print(f"\nScraping: {url}")

        results = scraper.scrape_results(url)

        if results:
            print("\n" + "=" * 80)
            print("RACE INFORMATION")
            print("=" * 80)
            print(f"Track: {results.get('track', 'N/A')}")
            print(f"Race Number: {results.get('race_number', 'N/A')}")
            print(f"Distance: {results.get('distance', 'N/A')}")
            print(f"Grade: {results.get('grade', 'N/A')}")
            print(f"Prize Money: {results.get('prize_money', 'N/A')}")
            print(f"Race Time: {results.get('race_time', 'N/A')}")

            print("\n" + "=" * 80)
            print(f"RESULTS ({len(results.get('results', []))} entries)")
            print("=" * 80)

            for entry in results.get('results', []):
                print(f"\n{entry.get('position', '?')}. {entry.get('greyhound_name', 'Unknown')}")
                print(f"   Box: {entry.get('box', 'N/A')}")
                print(f"   Trainer: {entry.get('trainer', 'N/A')}")
                print(f"   Time: {entry.get('finish_time', 'N/A')}")
                print(f"   Margin: {entry.get('margin', 'N/A')}")
                print(f"   Split: {entry.get('split', 'N/A')}")
                print(f"   In Run: {entry.get('in_run', 'N/A')}")
                print(f"   Weight: {entry.get('weight', 'N/A')}")
                print(f"   Breeding: {entry.get('sire', 'N/A')} x {entry.get('dam', 'N/A')}")
                print(f"   SP: {entry.get('starting_price', 'N/A')}")

            print("\n" + "=" * 80)
            print("SUCCESS - Results scraped successfully!")
            print("=" * 80)
        else:
            print("\n[ERROR] No results returned from scraper")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()


if __name__ == "__main__":
    test_full_results_scraper()
