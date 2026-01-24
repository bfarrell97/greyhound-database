"""Debug script to see what headers are in the results table"""

from greyhound_scraper_v2 import GreyhoundScraper
from bs4 import BeautifulSoup
import time

url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"

scraper = GreyhoundScraper(headless=False)
scraper.setup_driver()
scraper.driver.get(url)

time.sleep(5)

soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

table = soup.find('table')
if table:
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    print("Headers found:")
    for i, h in enumerate(headers):
        print(f"  {i}: '{h}'")

    print("\nFirst data row:")
    rows = table.find_all('tr')[1:]
    if rows:
        cols = rows[0].find_all('td')
        for i, col in enumerate(cols):
            print(f"  {i}: '{col.get_text(strip=True)}'")

    print("\nRow with DNF (if exists):")
    for row in rows:
        cols = row.find_all('td')
        if cols:
            first_col = cols[0].get_text(strip=True)
            if 'DNF' in first_col.upper():
                for i, col in enumerate(cols):
                    print(f"  {i}: '{col.get_text(strip=True)}'")
                break

    print("\nRow with SCR (if exists):")
    for row in rows:
        cols = row.find_all('td')
        if cols:
            first_col = cols[0].get_text(strip=True)
            if 'SCR' in first_col.upper():
                for i, col in enumerate(cols):
                    print(f"  {i}: '{col.get_text(strip=True)}'")
                break
else:
    print("No table found!")
