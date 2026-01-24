"""
Batch Scraper for Greyhound Racing Data
Allows scraping multiple URLs and saving to database
"""

from greyhound_scraper_v2 import GreyhoundScraper
from greyhound_database import GreyhoundDatabase
from datetime import datetime
import time


def scrape_and_save(url, race_date, db):
    """
    Scrape a single URL and save to database

    Args:
        url: Form guide or results URL
        race_date: Date string (YYYY-MM-DD)
        db: GreyhoundDatabase instance
    """
    scraper = GreyhoundScraper(headless=False)

    try:
        # Extract track name from URL
        parts = url.split('/')
        if '/form-guides/' in url:
            idx = parts.index('form-guides')
            track_slug = parts[idx + 1]
            track_name = ' '.join(word.capitalize() for word in track_slug.split('-'))

            print(f"\n[Form Guide] Scraping: {track_name}")
            data = scraper.scrape_form_guide(url)

            if data and data.get('entries'):
                print(f"  Found {len(data['entries'])} entries")
                success = db.import_form_guide_data(data, race_date, track_name)
                if success:
                    print(f"  SUCCESS - Saved to database")
                    return True
                else:
                    print(f"  ERROR - Failed to save")
                    return False
            else:
                print(f"  ERROR - No data scraped")
                return False

        elif '/results/' in url:
            idx = parts.index('results')
            track_slug = parts[idx + 1]
            track_name = ' '.join(word.capitalize() for word in track_slug.split('-'))

            print(f"\n[Results] Scraping: {track_name}")
            data = scraper.scrape_results(url)

            if data and data.get('results'):
                print(f"  Found {len(data['results'])} results")
                success = db.import_results_data(data, race_date, track_name)
                if success:
                    print(f"  SUCCESS - Saved to database")
                    return True
                else:
                    print(f"  ERROR - Failed to save")
                    return False
            else:
                print(f"  ERROR - No data scraped")
                return False
        else:
            print(f"  ERROR - Unknown URL type")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        scraper.close_driver()


def batch_scrape_from_file(filename='urls.txt'):
    """
    Scrape multiple URLs from a text file

    File format (one entry per line):
    URL,DATE

    Example:
    https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/,2025-11-29
    https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/,2025-12-01
    """
    db = GreyhoundDatabase('greyhound_racing.db')

    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print("=" * 80)
        print(f"BATCH SCRAPER - {len(lines)} URLs to process")
        print("=" * 80)

        success_count = 0
        fail_count = 0

        for i, line in enumerate(lines, 1):
            if ',' not in line:
                print(f"\n[{i}/{len(lines)}] Skipping invalid line: {line}")
                fail_count += 1
                continue

            url, date = line.split(',', 1)
            url = url.strip()
            date = date.strip()

            print(f"\n[{i}/{len(lines)}] Processing: {url}")
            print(f"  Date: {date}")

            if scrape_and_save(url, date, db):
                success_count += 1
            else:
                fail_count += 1

            # Small delay between scrapes to be polite
            if i < len(lines):
                print("  Waiting 3 seconds before next scrape...")
                time.sleep(3)

        print("\n" + "=" * 80)
        print("BATCH SCRAPE COMPLETE")
        print("=" * 80)
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Total: {len(lines)}")
        print("=" * 80)

    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found")
        print(f"\nCreate a file called '{filename}' with URLs and dates:")
        print("Example format:")
        print("https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/,2025-11-29")
        print("https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/,2025-12-01")
    finally:
        db.close()


def interactive_scrape():
    """Interactive mode - prompts for URL and date"""
    db = GreyhoundDatabase('greyhound_racing.db')

    print("=" * 80)
    print("INTERACTIVE SCRAPER")
    print("=" * 80)
    print("\nEnter URLs to scrape (or 'quit' to exit)")
    print("Format: Paste URL, then enter date when prompted\n")

    try:
        while True:
            print("-" * 80)
            url = input("URL (or 'quit'): ").strip()

            if url.lower() in ['quit', 'exit', 'q']:
                break

            if not url:
                continue

            date = input("Date (YYYY-MM-DD): ").strip()

            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
                print(f"  Using today's date: {date}")

            scrape_and_save(url, date, db)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        db.close()

    print("\n" + "=" * 80)
    print("SCRAPING SESSION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("GREYHOUND RACING BATCH SCRAPER")
    print("=" * 80)
    print("\nModes:")
    print("  1. Interactive mode (enter URLs one by one)")
    print("  2. Batch mode (read from urls.txt file)")
    print("  3. Single URL mode")
    print("=" * 80)

    if len(sys.argv) > 1:
        # Command line arguments provided
        if sys.argv[1] == 'batch':
            filename = sys.argv[2] if len(sys.argv) > 2 else 'urls.txt'
            batch_scrape_from_file(filename)
        elif sys.argv[1] == 'interactive':
            interactive_scrape()
        else:
            # Assume it's a URL
            url = sys.argv[1]
            date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")
            db = GreyhoundDatabase('greyhound_racing.db')
            scrape_and_save(url, date, db)
            db.close()
    else:
        # No arguments - show menu
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == '1':
            interactive_scrape()
        elif choice == '2':
            filename = input("Filename (default: urls.txt): ").strip()
            if not filename:
                filename = 'urls.txt'
            batch_scrape_from_file(filename)
        elif choice == '3':
            url = input("URL: ").strip()
            date = input("Date (YYYY-MM-DD): ").strip()
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            db = GreyhoundDatabase('greyhound_racing.db')
            scrape_and_save(url, date, db)
            db.close()
        else:
            print("Invalid choice")
