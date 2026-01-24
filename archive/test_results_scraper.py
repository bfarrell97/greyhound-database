"""
Test Results Page Scraping
"""

from greyhound_scraper_v2 import GreyhoundScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time as time_module
import re


def test_results_page():
    """Test scraping results page"""
    print("=" * 80)
    print("Testing Results Page Scraping")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        # Use the results URL you provided
        url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"
        print(f"\nNavigating to: {url}")

        scraper.setup_driver()
        scraper.driver.get(url)

        # Wait for page to load
        try:
            wait = WebDriverWait(scraper.driver, 15)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            print("[OK] Page loaded successfully")
        except:
            print("[WARNING] Timeout waiting for table")

        time_module.sleep(3)

        # Save page source
        with open("results_page.html", "w", encoding="utf-8") as f:
            f.write(scraper.driver.page_source)
        print("[OK] Saved page source to: results_page.html")

        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # ==================== EXTRACT RACE INFORMATION ====================
        print("\n" + "-" * 80)
        print("RACE INFORMATION:")
        print("-" * 80)

        # Find page title
        h1 = soup.find('h1')
        if h1:
            print(f"Page Title: {h1.get_text(strip=True)}")

        # Find race number buttons
        print("\n" + "-" * 80)
        print("RACE NUMBERS:")
        print("-" * 80)

        buttons = soup.find_all('button')
        race_buttons = [b for b in buttons if b.get_text(strip=True).isdigit()]
        print(f"Found {len(race_buttons)} race buttons: {', '.join(b.get_text(strip=True) for b in race_buttons[:12])}")

        # ==================== EXTRACT RESULTS TABLE ====================
        print("\n" + "-" * 80)
        print("RESULTS TABLE:")
        print("-" * 80)

        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables on page")

        for i, table in enumerate(tables, 1):
            print(f"\n--- Table {i} ---")

            # Get headers
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            print(f"Headers: {headers}")

            # Get first 3 data rows
            rows = table.find_all('tr')[1:]  # Skip header
            print(f"Data rows: {len(rows)}")

            if rows:
                print(f"\nFirst 3 rows:")
                for j, row in enumerate(rows[:3], 1):
                    cols = [td.get_text(strip=True) for td in row.find_all('td')]
                    print(f"  Row {j}: {cols}")

        # Look for specific result data
        print("\n" + "-" * 80)
        print("SEARCHING FOR SPECIFIC ELEMENTS:")
        print("-" * 80)

        # Look for race event header
        event_header = soup.find('div', class_='meeting-event__header')
        if event_header:
            print("\nFound meeting event header:")
            # Distance
            dist_elem = event_header.find(class_='meeting-event__header-distance')
            if dist_elem:
                print(f"  Distance: {dist_elem.get_text(strip=True)}")

            # Class/Grade
            class_elem = event_header.find(class_='meeting-event__header-class')
            if class_elem:
                print(f"  Class: {class_elem.get_text(strip=True)}")

            # Prize
            prize_elem = event_header.find(class_='meeting-event__header-prize')
            if prize_elem:
                print(f"  Prize: {prize_elem.get_text(strip=True)}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing browser in 5 seconds...")
        time_module.sleep(5)
        scraper.close_driver()


if __name__ == "__main__":
    test_results_page()
