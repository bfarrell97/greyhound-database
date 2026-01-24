import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.integration.scraper_v2 import GreyhoundScraper
import datetime

def debug_meetings():
    print("Initializing Scraper...")
    scraper = GreyhoundScraper(headless=True) # Try headless first, maybe switch if fails
    
    print("Navigating to Form Guides...")
    scraper.setup_driver()
    scraper.driver.get("https://www.thegreyhoundrecorder.com.au/form-guides/")
    
    print("Page Title:", scraper.driver.title)
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')
    
    # Debug generic meeting rows
    rows = soup.find_all('div', class_='meeting-row')
    print(f"Found {len(rows)} elements with class 'meeting-row'")
    
    for i, row in enumerate(rows):
        text = row.get_text(strip=True)
        link = row.find('a', href=True)
        href = link['href'] if link else "No Link"
        print(f"Row {i+1}: {text[:50]}... -> {href}")
        
    scraper.close_driver()

if __name__ == "__main__":
    debug_meetings()
