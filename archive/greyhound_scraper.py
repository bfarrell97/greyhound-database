"""
Greyhound Racing Web Scraper
Scrapes race data from The Greyhound Recorder website
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime


class GreyhoundScraper:
    """Web scraper for greyhound racing data"""

    def __init__(self, headless=True):
        """Initialize the scraper with Chrome driver"""
        self.headless = headless
        self.driver = None
        self.base_url = "https://www.thegreyhoundrecorder.com.au"

    def setup_driver(self):
        """Set up Chrome driver"""
        if self.driver:
            return

        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def close_driver(self):
        """Close the Chrome driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def get_upcoming_meetings(self, date_str=None):
        """
        Get list of upcoming race meetings
        date_str format: 'YYYY-MM-DD' or None for today
        Returns: List of meeting data with links
        """
        self.setup_driver()

        if date_str:
            # Convert to desired format if needed
            url = f"{self.base_url}/form-guides/"
        else:
            url = f"{self.base_url}/form-guides/"

        self.driver.get(url)
        time.sleep(2)

        # Parse the page
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        meetings = []

        # Find all track cards/buttons
        # Based on screenshots, tracks are shown with name and country code
        # This will need to be refined based on actual HTML structure
        track_elements = soup.find_all('a', href=re.compile(r'/form-guides/'))

        for track in track_elements:
            href = track.get('href', '')
            if '/long-form/' in href or '/short-form/' in href or '/fields/' in href:
                # Extract track name from href or text
                track_name = track.get_text(strip=True)

                # Try to get the track key from href
                match = re.search(r'/form-guides/([^/]+)/', href)
                if match:
                    track_key = match.group(1)

                    meetings.append({
                        'track_name': track_name,
                        'track_key': track_key,
                        'url': f"{self.base_url}{href}" if not href.startswith('http') else href
                    })

        return meetings

    def get_race_links_for_track(self, track_key, date_str=None):
        """
        Get all race links for a specific track
        Returns: List of race URLs with their IDs
        """
        self.setup_driver()

        # Go to track page
        url = f"{self.base_url}/form-guides/{track_key}/long-form/"
        self.driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        race_links = []

        # Find all long-form race links
        # Pattern: /form-guides/{track}/long-form/{race_id}/{race_number}/
        links = soup.find_all('a', href=re.compile(r'/form-guides/.+/long-form/\d+/\d+'))

        for link in links:
            href = link.get('href', '')
            # Extract race ID and number from URL
            match = re.search(r'/long-form/(\d+)/(\d+)', href)
            if match:
                race_id = match.group(1)
                race_number = match.group(2)

                race_links.append({
                    'url': f"{self.base_url}{href}" if not href.startswith('http') else href,
                    'race_id': race_id,
                    'race_number': race_number,
                    'track_key': track_key
                })

        return race_links

    def scrape_race_form(self, race_url):
        """
        Scrape detailed race form data from long-form URL
        Returns: Dictionary with race and greyhound data
        """
        self.setup_driver()
        self.driver.get(race_url)
        time.sleep(2)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        race_data = {
            'entries': [],
            'race_info': {}
        }

        # Extract race information from header
        # Race header shows: "Broken Hill Form Guide (Race 1) - 30th Nov 2025"
        # And details like: "LADBROKES QUICK MULTI 0-2 WIN  375m  XM5  $1,475 - $435 - $325"

        try:
            # Get race header
            race_header = soup.find('h1')
            if race_header:
                header_text = race_header.get_text(strip=True)
                # Extract track and race number
                match = re.search(r'(.+?)\s+Form Guide\s+\(Race\s+(\d+)\)\s+-\s+(.+)', header_text)
                if match:
                    race_data['track_name'] = match.group(1)
                    race_data['race_number'] = int(match.group(2))
                    race_data['date'] = match.group(3)

            # Get race details (grade, distance, prize money, time)
            race_details = soup.find('div', class_=re.compile(r'race.*info|Race.*'))
            if race_details:
                details_text = race_details.get_text()

                # Extract distance (e.g., "375m")
                dist_match = re.search(r'(\d+)m', details_text)
                if dist_match:
                    race_data['distance'] = int(dist_match.group(1))

                # Extract grade (e.g., "XM5", "A1 SIGNAGE")
                grade_match = re.search(r'(XM\d+|A\d+|M\d+|[A-Z]\d+)', details_text)
                if grade_match:
                    race_data['grade'] = grade_match.group(1)

                # Extract prize money
                prize_match = re.search(r'\$[\d,]+ - \$[\d,]+ - \$[\d,]+', details_text)
                if prize_match:
                    race_data['prize_money'] = prize_match.group(0)

                # Extract race time
                time_match = re.search(r'(\d+:\d+(?:AM|PM))', details_text)
                if time_match:
                    race_data['race_time'] = time_match.group(1)

        except Exception as e:
            print(f"Error extracting race header: {e}")

        # Now scrape each greyhound's data
        # Look for greyhound cards/rows
        greyhound_sections = soup.find_all('div', class_=re.compile(r'greyhound|runner|entry'))

        if not greyhound_sections:
            # Alternative: look for table rows
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                greyhound_sections = rows

        for section in greyhound_sections:
            try:
                entry = self.parse_greyhound_entry(section)
                if entry:
                    race_data['entries'].append(entry)
            except Exception as e:
                print(f"Error parsing greyhound entry: {e}")
                continue

        return race_data

    def parse_greyhound_entry(self, section):
        """Parse a single greyhound entry from HTML"""
        entry = {}

        try:
            # Extract greyhound name
            name_elem = section.find(text=re.compile(r'[A-Z\s]{3,}'))
            if name_elem:
                entry['greyhound_name'] = name_elem.strip()

            # Extract box number
            box_elem = section.find(class_=re.compile(r'box|rug'))
            if box_elem:
                box_text = box_elem.get_text(strip=True)
                box_match = re.search(r'(\d+)', box_text)
                if box_match:
                    entry['box'] = int(box_match.group(1))

            # Extract trainer
            trainer_elem = section.find(text=re.compile(r'Trainer:'))
            if trainer_elem:
                trainer_text = trainer_elem.parent.get_text()
                match = re.search(r'Trainer:\s*(.+?)(?:\(|$)', trainer_text)
                if match:
                    entry['trainer'] = match.group(1).strip()

            # Extract owner
            owner_elem = section.find(text=re.compile(r'Owner:'))
            if owner_elem:
                owner_text = owner_elem.parent.get_text()
                match = re.search(r'Owner:\s*(.+)', owner_text)
                if match:
                    entry['owner'] = match.group(1).strip()

            # Extract breeding info
            breeding = section.find(text=re.compile(r'[A-Z]+\s*/\s*D\s+'))
            if breeding:
                # Pattern: "RF / D Mar-24 Sire x Dam"
                sire_dam = re.search(r'(.+?)\s+x\s+(.+)', breeding)
                if sire_dam:
                    entry['sire'] = sire_dam.group(1).strip()
                    entry['dam'] = sire_dam.group(2).strip()

            # Extract form (e.g., "1173", "5286")
            form_elem = section.find(class_=re.compile(r'form'))
            if form_elem:
                entry['form'] = form_elem.get_text(strip=True)

            # Extract stats
            # Prize money, best times, ratings, etc.
            stats_text = section.get_text()

            # Prize money
            prize_match = re.search(r'Prizemoney:\s*\$?([\d,]+)', stats_text)
            if prize_match:
                entry['prizemoney'] = float(prize_match.group(1).replace(',', ''))

            # Rating
            rating_match = re.search(r'Rating:\s*(\d+)', stats_text)
            if rating_match:
                entry['rating'] = int(rating_match.group(1))

            # Best time
            best_time_match = re.search(r'Best(?:\s+Win)?\s+Times?:\s*\w+\s+([\d.]+)', stats_text)
            if best_time_match:
                entry['best_time'] = float(best_time_match.group(1))

            # Weight
            weight_match = re.search(r'(\d+\.?\d*)\s*kg', stats_text)
            if weight_match:
                entry['weight'] = float(weight_match.group(1))

            # Starting price
            price_match = re.search(r'Our Price:\s*\$?([\d.]+)', stats_text)
            if price_match:
                entry['starting_price'] = price_match.group(1)

        except Exception as e:
            print(f"Error in parse_greyhound_entry: {e}")

        return entry if entry else None

    def get_results_meetings(self, date_str=None):
        """
        Get list of race meetings with results
        date_str format: 'YYYY-MM-DD' or None for latest
        Returns: List of meeting data with result links
        """
        self.setup_driver()

        url = f"{self.base_url}/results/"
        if date_str:
            url += f"?date={date_str}"

        self.driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        meetings = []

        # Find all result links
        # Pattern: /results/{track}/{meeting_id}/
        links = soup.find_all('a', href=re.compile(r'/results/[^/]+/\d+'))

        for link in links:
            href = link.get('href', '')
            match = re.search(r'/results/([^/]+)/(\d+)', href)
            if match:
                track_key = match.group(1)
                meeting_id = match.group(2)
                track_name = link.get_text(strip=True)

                meetings.append({
                    'track_key': track_key,
                    'track_name': track_name,
                    'meeting_id': meeting_id,
                    'url': f"{self.base_url}{href}" if not href.startswith('http') else href
                })

        return meetings

    def scrape_race_results(self, results_url):
        """
        Scrape race results from results page
        Returns: Dictionary with race results
        """
        self.setup_driver()
        self.driver.get(results_url)
        time.sleep(2)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        race_data = {
            'entries': [],
            'race_info': {}
        }

        try:
            # Extract race information
            # Similar structure to form guide but with actual results

            # Get race header
            race_header = soup.find('h1')
            if race_header:
                header_text = race_header.get_text(strip=True)
                # Extract track name
                match = re.search(r'(.+?)\s+Race Results', header_text)
                if match:
                    race_data['track_name'] = match.group(1)

            # Find results table
            table = soup.find('table')
            if table:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        continue

                    entry = {}

                    # Map columns based on headers
                    # Typical columns: Plc, Rug, Name (Box), Trainer, Time, Mgn, Split, In Run, Wgt, Sire, Dam, SP
                    for i, col in enumerate(cols):
                        col_text = col.get_text(strip=True)

                        if i < len(headers):
                            header = headers[i].lower()

                            if 'plc' in header or 'place' in header:
                                entry['position'] = int(col_text) if col_text.isdigit() else None
                            elif 'name' in header or 'box' in header:
                                # Extract greyhound name and box
                                name_match = re.search(r'(.+?)\s*\((\d+)\)', col_text)
                                if name_match:
                                    entry['greyhound_name'] = name_match.group(1).strip()
                                    entry['box'] = int(name_match.group(2))
                                else:
                                    entry['greyhound_name'] = col_text
                            elif 'trainer' in header:
                                entry['trainer'] = col_text
                            elif 'time' in header:
                                time_match = re.search(r'([\d.]+)', col_text)
                                if time_match:
                                    entry['finish_time'] = float(time_match.group(1))
                            elif 'mgn' in header or 'margin' in header:
                                margin_match = re.search(r'([\d.]+)', col_text)
                                if margin_match:
                                    entry['margin'] = float(margin_match.group(1))
                            elif 'split' in header:
                                split_match = re.search(r'([\d.]+)', col_text)
                                if split_match:
                                    entry['split'] = float(split_match.group(1))
                            elif 'wgt' in header or 'weight' in header:
                                weight_match = re.search(r'([\d.]+)', col_text)
                                if weight_match:
                                    entry['weight'] = float(weight_match.group(1))
                            elif 'sire' in header:
                                entry['sire'] = col_text
                            elif 'dam' in header:
                                entry['dam'] = col_text
                            elif 'sp' in header or 'price' in header:
                                entry['starting_price'] = col_text

                    if entry:
                        race_data['entries'].append(entry)

        except Exception as e:
            print(f"Error scraping results: {e}")

        return race_data

    def scrape_meeting(self, meeting_url, is_results=False):
        """
        Scrape all races from a meeting
        is_results: True for historical results, False for form guides
        """
        self.setup_driver()
        self.driver.get(meeting_url)
        time.sleep(2)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        races = []

        # Find all race number buttons/links (1, 2, 3, 4, etc.)
        race_buttons = soup.find_all('button', text=re.compile(r'^\d+$'))
        if not race_buttons:
            race_buttons = soup.find_all('a', text=re.compile(r'^\d+$'))

        for button in race_buttons:
            race_num = button.get_text(strip=True)
            if race_num.isdigit():
                # Click or navigate to this race
                # Since we're using static scraping, we'll need to construct URLs
                # This will need refinement based on actual site behavior
                pass

        return races


if __name__ == "__main__":
    # Example usage
    scraper = GreyhoundScraper(headless=False)

    try:
        # Get upcoming meetings
        print("Fetching upcoming meetings...")
        meetings = scraper.get_upcoming_meetings()

        for meeting in meetings[:3]:  # First 3 meetings
            print(f"\nMeeting: {meeting['track_name']}")
            print(f"URL: {meeting['url']}")

        # Scrape a specific race
        if meetings:
            print(f"\nScraping first meeting...")
            # Get race links for first track
            race_links = scraper.get_race_links_for_track(meetings[0]['track_key'])

            if race_links:
                print(f"Found {len(race_links)} races")
                # Scrape first race
                race_data = scraper.scrape_race_form(race_links[0]['url'])
                print(f"\nRace data:")
                print(f"Track: {race_data.get('track_name')}")
                print(f"Race Number: {race_data.get('race_number')}")
                print(f"Distance: {race_data.get('distance')}m")
                print(f"Entries: {len(race_data.get('entries', []))}")

    finally:
        scraper.close_driver()
