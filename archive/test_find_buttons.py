"""
Debug script to find the race selector buttons
"""

from greyhound_scraper_v2 import GreyhoundScraper
from selenium.webdriver.common.by import By
import time

def debug_buttons():
    """Find and print info about race selector buttons"""

    meeting_url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"

    print("=" * 80)
    print("DEBUGGING RACE SELECTOR BUTTONS")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        scraper.setup_driver()
        scraper.driver.get(meeting_url)

        # Wait for page to load
        time.sleep(5)

        # Save page source to file for inspection
        with open('page_source.html', 'w', encoding='utf-8') as f:
            f.write(scraper.driver.page_source)
        print("\nPage source saved to page_source.html")

        # Try different selectors
        selectors = [
            "button.meeting-race-number-selector__button",
            "button[class*='race-number']",
            "button[class*='selector']",
            ".meeting-race-number-selector button",
            "div.meeting-race-number-selector button",
            "button"
        ]

        for selector in selectors:
            print(f"\nTrying selector: {selector}")
            elements = scraper.driver.find_elements(By.CSS_SELECTOR, selector)
            print(f"  Found {len(elements)} elements")

            if elements:
                for i, elem in enumerate(elements[:5], 1):  # Show first 5
                    print(f"    [{i}] Text: '{elem.text}' | Class: '{elem.get_attribute('class')}'")

        # Also try finding by text
        print("\nSearching for buttons with text '1', '2', '3', etc...")
        all_buttons = scraper.driver.find_elements(By.TAG_NAME, "button")
        race_buttons = [b for b in all_buttons if b.text.strip().isdigit()]
        print(f"Found {len(race_buttons)} buttons with digit text")

        for i, btn in enumerate(race_buttons[:12], 1):
            print(f"  [{i}] Text: '{btn.text}' | Class: '{btn.get_attribute('class')}'")

    finally:
        input("\nPress Enter to close browser...")
        scraper.close_driver()

if __name__ == "__main__":
    debug_buttons()
