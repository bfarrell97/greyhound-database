"""Debug script to click Race 4 and see the data"""

from greyhound_scraper_v2 import GreyhoundScraper
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time

url = "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/"

scraper = GreyhoundScraper(headless=False)
scraper.setup_driver()
scraper.driver.get(url)

time.sleep(5)

# Click on Race 4 button
print("Looking for Race 4 button...")
race_buttons = scraper.driver.find_elements(By.CSS_SELECTOR, ".meeting-events-nav__item")
print(f"Found {len(race_buttons)} race buttons")

if len(race_buttons) >= 4:
    race_4_button = race_buttons[3]  # 0-indexed, so 3 is race 4
    print("Clicking Race 4...")
    scraper.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", race_4_button)
    time.sleep(0.5)
    race_4_button.click()
    time.sleep(3)  # Wait for race 4 to load

    soup = BeautifulSoup(scraper.driver.page_source, 'html.parser')

    table = soup.find('table')
    if table:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        print("\nHeaders found:")
        for i, h in enumerate(headers):
            print(f"  {i}: '{h}'")

        print("\nAll rows:")
        rows = table.find_all('tr')[1:]
        for row_num, row in enumerate(rows, 1):
            cols = row.find_all('td')
            if cols:
                plc = cols[0].get_text(strip=True)
                name_box = cols[2].get_text(strip=True) if len(cols) > 2 else ''
                print(f"  Row {row_num}: Plc='{plc}' Name='{name_box}'")
    else:
        print("No table found!")
else:
    print("Not enough race buttons found!")

input("\nPress Enter to close browser...")
