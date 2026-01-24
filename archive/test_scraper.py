"""
Test Scraper Script
Tests and debugs the greyhound scraper
"""

from greyhound_scraper import GreyhoundScraper
from bs4 import BeautifulSoup
import time


def test_upcoming_meetings():
    """Test scraping upcoming meetings list"""
    print("=" * 80)
    print("TEST 1: Scraping Upcoming Meetings List")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        url = "https://www.thegreyhoundrecorder.com.au/form-guides/"
        print(f"\nNavigating to: {url}")

        scraper.setup_driver()
        scraper.driver.get(url)
        time.sleep(3)  # Wait for page to load

        # Save page source for inspection
        with open("form_guides_page.html", "w", encoding="utf-8") as f:
            f.write(scraper.driver.page_source)
        print("✓ Saved page source to: form_guides_page.html")

        # Parse with BeautifulSoup
        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # Look for track/meeting links
        print("\n" + "-" * 80)
        print("Looking for meeting links...")
        print("-" * 80)

        # Find all links
        all_links = soup.find_all('a')
        print(f"\nFound {len(all_links)} total links on page")

        # Filter for form guide links
        form_links = [link for link in all_links if link.get('href') and '/form-guides/' in link.get('href')]
        print(f"Found {len(form_links)} form guide links")

        # Show first 10 form guide links
        print("\nFirst 10 form guide links:")
        for i, link in enumerate(form_links[:10], 1):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            print(f"{i}. {text[:40]:40} -> {href}")

        print("\n" + "=" * 80)
        input("Press Enter to continue to next test...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()


def test_race_card():
    """Test scraping a specific race card"""
    print("\n" + "=" * 80)
    print("TEST 2: Scraping a Specific Race Card")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        # Use the example URL you provided
        url = "https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/"
        print(f"\nNavigating to: {url}")

        scraper.setup_driver()
        scraper.driver.get(url)
        time.sleep(3)

        # Save page source
        with open("race_card_page.html", "w", encoding="utf-8") as f:
            f.write(scraper.driver.page_source)
        print("✓ Saved page source to: race_card_page.html")

        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # Look for race information
        print("\n" + "-" * 80)
        print("Race Information:")
        print("-" * 80)

        # Find page title/header
        title = soup.find('h1')
        if title:
            print(f"Page Title: {title.get_text(strip=True)}")

        # Look for race details
        print("\nSearching for race details...")

        # Try to find race info section
        race_sections = soup.find_all(['div', 'section'], class_=lambda x: x and 'race' in x.lower())
        print(f"Found {len(race_sections)} elements with 'race' in class name")

        # Look for greyhound entries
        print("\n" + "-" * 80)
        print("Greyhound Entries:")
        print("-" * 80)

        # Find all headings that might be greyhound names
        all_text = soup.find_all(text=True)
        potential_names = [t.strip() for t in all_text if t.strip() and len(t.strip()) > 3 and t.strip().isupper()]

        print(f"\nFound {len(potential_names)} potential greyhound names (all caps text):")
        for i, name in enumerate(potential_names[:20], 1):
            print(f"{i}. {name[:50]}")

        # Look for table structure
        tables = soup.find_all('table')
        print(f"\n\nFound {len(tables)} tables on page")

        if tables:
            for i, table in enumerate(tables, 1):
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows = len(table.find_all('tr'))
                print(f"\nTable {i}: {rows} rows")
                if headers:
                    print(f"  Headers: {', '.join(headers[:10])}")

        print("\n" + "=" * 80)
        input("Press Enter to continue to next test...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()


def test_results_page():
    """Test scraping results page"""
    print("\n" + "=" * 80)
    print("TEST 3: Scraping Results Page")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        # Use the results URL you provided
        url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
        print(f"\nNavigating to: {url}")

        scraper.setup_driver()
        scraper.driver.get(url)
        time.sleep(3)

        # Save page source
        with open("results_page.html", "w", encoding="utf-8") as f:
            f.write(scraper.driver.page_source)
        print("✓ Saved page source to: results_page.html")

        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # Look for results table
        print("\n" + "-" * 80)
        print("Results Table:")
        print("-" * 80)

        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")

        if tables:
            for i, table in enumerate(tables, 1):
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows = table.find_all('tr')[1:]  # Skip header row

                print(f"\nTable {i}:")
                print(f"  Headers: {headers}")
                print(f"  Data rows: {len(rows)}")

                if rows:
                    print(f"\n  First row data:")
                    first_row = rows[0]
                    cells = [td.get_text(strip=True) for td in first_row.find_all('td')]
                    for j, (header, cell) in enumerate(zip(headers, cells)):
                        print(f"    {header}: {cell}")

        # Look for race number buttons
        print("\n" + "-" * 80)
        print("Race Number Buttons:")
        print("-" * 80)

        buttons = soup.find_all('button')
        print(f"Found {len(buttons)} buttons")

        race_buttons = [b for b in buttons if b.get_text(strip=True).isdigit()]
        print(f"Found {len(race_buttons)} numeric buttons (likely race numbers)")
        if race_buttons:
            print(f"Race numbers: {', '.join(b.get_text(strip=True) for b in race_buttons[:15])}")

        print("\n" + "=" * 80)
        print("Test complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_driver()


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GREYHOUND SCRAPER TEST SUITE")
    print("=" * 80)
    print("\nThis will open Chrome browser windows to test scraping.")
    print("HTML files will be saved for inspection.")
    print("\n" + "=" * 80)

    input("Press Enter to start tests...")

    # Run tests one by one
    test_upcoming_meetings()
    test_race_card()
    test_results_page()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - form_guides_page.html")
    print("  - race_card_page.html")
    print("  - results_page.html")
    print("\nReview these files to understand the HTML structure.")
    print("=" * 80)


if __name__ == "__main__":
    main()
