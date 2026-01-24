"""
Automatic Scraper Test Script
Tests the greyhound scraper automatically
"""

from greyhound_scraper import GreyhoundScraper
from bs4 import BeautifulSoup
import time


def test_race_card():
    """Test scraping a specific race card"""
    print("=" * 80)
    print("Testing Race Card Scraping")
    print("=" * 80)

    scraper = GreyhoundScraper(headless=False)

    try:
        # Use the example URL you provided
        url = "https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/"
        print(f"\nNavigating to: {url}")
        print("Please wait...")

        scraper.setup_driver()
        scraper.driver.get(url)
        time.sleep(5)  # Wait for page to load

        # Save page source
        with open("race_card_page.html", "w", encoding="utf-8") as f:
            f.write(scraper.driver.page_source)
        print("[OK] Saved page source to: race_card_page.html")

        soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

        # Look for race information
        print("\n" + "-" * 80)
        print("RACE INFORMATION FOUND:")
        print("-" * 80)

        # Find page title/header
        h1_tags = soup.find_all('h1')
        print(f"\nH1 Tags ({len(h1_tags)}):")
        for h1 in h1_tags:
            print(f"  {h1.get_text(strip=True)}")

        # Look for race details in any element
        print(f"\n" + "-" * 80)
        print("SEARCHING FOR RACE DETAILS:")
        print("-" * 80)

        # Find distance (e.g., "375m")
        page_text = soup.get_text()
        import re

        distances = re.findall(r'\b(\d{3,4})m\b', page_text)
        if distances:
            print(f"Distances found: {set(distances)}")

        # Find grades
        grades = re.findall(r'\b([A-Z]{1,2}[A-Z0-9]{1,3})\b', page_text)
        potential_grades = [g for g in set(grades) if len(g) <= 5 and any(c.isdigit() for c in g)]
        print(f"Potential grades: {list(potential_grades)[:10]}")

        # Find prize money
        prize = re.findall(r'\$[\d,]+ - \$[\d,]+ - \$[\d,]+', page_text)
        if prize:
            print(f"Prize money: {prize[0]}")

        # Look for greyhound entries
        print("\n" + "-" * 80)
        print("SEARCHING FOR GREYHOUND DATA:")
        print("-" * 80)

        # Look for all divs (greyhounds are usually in divs)
        all_divs = soup.find_all('div')
        print(f"Total DIVs on page: {len(all_divs)}")

        # Look for specific patterns
        print("\nLooking for 'Trainer:' text...")
        trainer_texts = soup.find_all(text=re.compile(r'Trainer:', re.IGNORECASE))
        print(f"Found {len(trainer_texts)} instances of 'Trainer:'")

        if trainer_texts:
            for i, t in enumerate(trainer_texts[:3], 1):
                parent_text = t.parent.get_text()[:100]
                print(f"  {i}. {parent_text}")

        print("\nLooking for 'Owner:' text...")
        owner_texts = soup.find_all(text=re.compile(r'Owner:', re.IGNORECASE))
        print(f"Found {len(owner_texts)} instances of 'Owner:'")

        # Look for box numbers (rugs)
        print("\nLooking for box/rug numbers...")
        box_divs = soup.find_all(['div', 'span'], class_=re.compile(r'box|rug', re.IGNORECASE))
        print(f"Found {len(box_divs)} elements with 'box' or 'rug' in class")

        for i, box in enumerate(box_divs[:5], 1):
            print(f"  {i}. Class: {box.get('class')}, Text: {box.get_text(strip=True)[:30]}")

        # Look for tables
        print("\n" + "-" * 80)
        print("TABLES:")
        print("-" * 80)
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")

        for i, table in enumerate(tables, 1):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = len(table.find_all('tr'))
            print(f"\nTable {i}: {rows} rows")
            if headers:
                print(f"  Headers: {', '.join(headers)}")

        # Get all class names used on the page
        print("\n" + "-" * 80)
        print("COMMON CLASS NAMES (for reference):")
        print("-" * 80)
        all_classes = []
        for element in soup.find_all(True):
            if element.get('class'):
                all_classes.extend(element.get('class'))

        from collections import Counter
        class_counts = Counter(all_classes)
        print("\nTop 20 most common classes:")
        for cls, count in class_counts.most_common(20):
            print(f"  {cls}: {count}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE!")
        print("=" * 80)
        print("\nCheck 'race_card_page.html' for full HTML source")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing browser in 3 seconds...")
        time.sleep(3)
        scraper.close_driver()


if __name__ == "__main__":
    test_race_card()
