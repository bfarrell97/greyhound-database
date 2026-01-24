"""
Test data extraction from rendered page
"""

from greyhound_scraper import GreyhoundScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time as time_module
import re


def test_extract_greyhound_data():
    """Test extracting greyhound data from form guide"""
    print("=" * 80)
    print("Testing Greyhound Data Extraction")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        url = "https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/"
        print(f"\nNavigating to: {url}")

        scraper.setup_driver()
        scraper.driver.get(url)

        # Wait for page to load dynamically
        print("Waiting for page to load...")
        wait = WebDriverWait(scraper.driver, 15)

        # Wait for specific element that indicates page is loaded
        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "form-guide-long-form-selection")))
            print("[OK] Page loaded successfully")
        except:
            print("[WARNING] Timeout waiting for main content, continuing anyway...")

        time_module.sleep(3)  # Extra wait for dynamic content

        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # Find all greyhound selection divs
        print("\n" + "-" * 80)
        print("EXTRACTING GREYHOUND ENTRIES:")
        print("-" * 80)

        greyhound_divs = soup.find_all('div', class_='form-guide-long-form-selection')
        print(f"\nFound {len(greyhound_divs)} greyhound entries")

        for i, div in enumerate(greyhound_divs, 1):
            print(f"\n{'='*60}")
            print(f"GREYHOUND #{i}")
            print(f"{'='*60}")

            entry = {}

            # Extract greyhound name
            name_elem = div.find(class_='form-guide-long-form-selection__header-name')
            if name_elem:
                entry['name'] = name_elem.get_text(strip=True)
                print(f"Name: {entry['name']}")

            # Extract box number
            box_elem = div.find(class_='form-guide-long-form-selection__header-box')
            if box_elem:
                entry['box'] = box_elem.get_text(strip=True)
                print(f"Box: {entry['box']}")

            # Extract pedigree (sire x dam)
            pedigree_elem = div.find(class_='form-guide-long-form-selection__header-pedigree')
            if pedigree_elem:
                pedigree = pedigree_elem.get_text(strip=True)
                entry['pedigree'] = pedigree
                print(f"Pedigree: {pedigree}")

                # Try to split into sire and dam
                if ' x ' in pedigree:
                    parts = pedigree.split(' x ')
                    if len(parts) >= 2:
                        entry['sire'] = parts[0].strip()
                        entry['dam'] = parts[1].strip()
                        print(f"  Sire: {entry['sire']}")
                        print(f"  Dam: {entry['dam']}")

            # Extract trainer
            trainer_elem = div.find(class_='form-guide-long-form-selection__header-trainer')
            if trainer_elem:
                trainer_text = trainer_elem.get_text(strip=True)
                # Format is usually "Trainer: Name Owner: Name"
                trainer_match = re.search(r'Trainer:\s*(.+?)(?:Owner:|$)', trainer_text)
                if trainer_match:
                    entry['trainer'] = trainer_match.group(1).strip()
                    print(f"Trainer: {entry['trainer']}")

                owner_match = re.search(r'Owner:\s*(.+?)$', trainer_text)
                if owner_match:
                    entry['owner'] = owner_match.group(1).strip()
                    print(f"Owner: {entry['owner']}")

            # Extract prize money
            prize_elem = div.find(class_='form-guide-long-form-selection__header-prize')
            if prize_elem:
                prize_text = prize_elem.get_text(strip=True)
                prize_match = re.search(r'\$?([\d,]+)', prize_text)
                if prize_match:
                    entry['prizemoney'] = prize_match.group(1)
                    print(f"Prizemoney: ${entry['prizemoney']}")

            # Extract rating
            rating_elem = div.find(class_='form-guide-long-form-selection__header-rating-value')
            if rating_elem:
                entry['rating'] = rating_elem.get_text(strip=True)
                print(f"Rating: {entry['rating']}")

            # Extract best win times
            best_win_elem = div.find(class_='form-guide-long-form-selection__header-best-win')
            if best_win_elem:
                best_win_text = best_win_elem.get_text(strip=True)
                entry['best_win'] = best_win_text
                print(f"Best Win: {best_win_text}")

            # Extract historical form table
            print(f"\n  Historical Form:")
            form_table = div.find('table', class_='form-guide-selection-results')
            if form_table:
                rows = form_table.find_all('tr')[1:]  # Skip header
                print(f"  Found {len(rows)} historical races")

                for j, row in enumerate(rows[:3], 1):  # Show first 3
                    cols = row.find_all('td')
                    if len(cols) >= 10:
                        date = cols[0].get_text(strip=True)
                        fin = cols[1].get_text(strip=True)
                        box = cols[2].get_text(strip=True)
                        mgn = cols[3].get_text(strip=True)
                        trk = cols[4].get_text(strip=True)
                        dis = cols[5].get_text(strip=True)
                        grd = cols[6].get_text(strip=True)
                        time = cols[7].get_text(strip=True)
                        sect = cols[10].get_text(strip=True) if len(cols) > 10 else 'N/A'

                        print(f"    {j}. {date} - {trk} {dis} {grd} - Box:{box} Fin:{fin} Time:{time} Sect:{sect}")

            if i >= 3:  # Only show first 3 greyhounds for testing
                print(f"\n... and {len(greyhound_divs) - 3} more greyhounds")
                break

        print("\n" + "=" * 80)
        print("EXTRACTION TEST COMPLETE!")
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
    test_extract_greyhound_data()
