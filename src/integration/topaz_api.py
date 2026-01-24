"""
Topaz API Client for Greyhound Racing Victoria
Provides access to greyhound racing data via the GRV Topaz API
"""

import requests
from datetime import datetime
from typing import Optional, List, Dict, Any


class TopazAPI:
    """Client for interacting with the GRV Topaz API"""

    # Track code to state mapping (common tracks)
    TRACK_STATE_MAP = {
        # Victoria
        'BEN': 'VIC', 'GEL': 'VIC', 'HOR': 'VIC', 'MEA': 'VIC', 'BAL': 'VIC',
        'SHP': 'VIC', 'SLE': 'VIC', 'WBL': 'VIC', 'WGL': 'VIC', 'CRA': 'VIC',
        'TAR': 'VIC', 'MEP': 'VIC', 'HVL': 'VIC', 'SAL': 'VIC',
        # NSW
        'BUL': 'NSW', 'DAB': 'NSW', 'GOS': 'NSW', 'LIS': 'NSW', 'RIC': 'NSW',
        'TEM': 'NSW', 'TWE': 'NSW', 'THE': 'NSW', 'WEN': 'NSW', 'MUL': 'NSW',
        'BUL': 'NSW', 'NEW': 'NSW',
        # QLD
        'ALB': 'QLD', 'CAI': 'QLD', 'IPS': 'QLD', 'TOW': 'QLD', 'BGC': 'QLD',
        # SA
        'ANG': 'SA', 'GAW': 'SA', 'MUR': 'SA', 'MTP': 'SA',
        # WA
        'CAN': 'WA', 'MAN': 'WA',
        # TAS
        'DEV': 'TAS', 'HOB': 'TAS', 'LAU': 'TAS',
        # NZ
        'WAN': 'NZ', 'ADK': 'NZ', 'HUT': 'NZ'
    }

    def __init__(self, api_key: str):
        """
        Initialize the Topaz API client

        Args:
            api_key: Your Topaz API key
        """
        self.api_key = api_key
        self.base_url = "https://topaz.grv.org.au/api"  # Added /api prefix
        self.headers = {
            "X-API-Key": api_key,
            "Accept": "application/json"
        }

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to the Topaz API

        Args:
            endpoint: API endpoint (e.g., '/codes/track')
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if 'response' in locals():
                print(f"Response: {response.text[:500]}")
            raise

    def get_track_codes(self) -> List[Dict[str, Any]]:
        """
        Get all track codes

        Returns:
            List of track dictionaries with trackId, trackCode, trackName, etc.
        """
        return self._make_request("/codes/track")

    def get_meetings(self, date_from: str, date_to: Optional[str] = None,
                     track_code: Optional[str] = None,
                     owning_authority_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get meetings for a date range and optional track

        Args:
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format (optional, defaults to date_from)
            track_code: Track code (e.g., 'MEA' for The Meadows) (optional)
            owning_authority_code: State code (e.g., 'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'NZ') (optional)

        Returns:
            List of meeting dictionaries
        """
        if date_to is None:
            date_to = date_from

        params = {
            "from": date_from,
            "to": date_to
        }

        if track_code:
            params["track"] = track_code

        if owning_authority_code:
            params["owningauthoritycode"] = owning_authority_code

        return self._make_request("/meeting", params)

    def get_races(self, meeting_id: int) -> List[Dict[str, Any]]:
        """
        Get all races for a specific meeting

        Args:
            meeting_id: Meeting ID from get_meetings()

        Returns:
            List of race dictionaries
        """
        return self._make_request(f"/meeting/{meeting_id}/races")

    def get_race_splits(self, race_id: int) -> Dict[str, Any]:
        """
        Get detailed split/runner data for a race

        Args:
            race_id: Race ID from get_races()

        Returns:
            Dictionary containing race details and runner splits data
        """
        return self._make_request(f"/isolynx/{race_id}/splits")

    def get_form_guide_data(self, date: str, track_code: str, race_number: int) -> Dict[str, Any]:
        """
        Get comprehensive form guide data for a specific race

        Args:
            date: Date in YYYY-MM-DD format
            track_code: Track code (e.g., 'MEA')
            race_number: Race number (1-12)

        Returns:
            Dictionary with race and runner details
        """
        # Step 1: Try to get meeting from known state first
        meeting = None

        # Check if we know which state this track is in
        if track_code in self.TRACK_STATE_MAP:
            state = self.TRACK_STATE_MAP[track_code]
            try:
                meetings = self.get_meetings(date, owning_authority_code=state)
                for m in meetings:
                    if m.get('trackCode') == track_code:
                        meeting = m
                        break
            except:
                pass

        # If not found, fall back to searching all states
        if not meeting:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import sys
            import io

            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']
            all_meetings = []

            def fetch_state_meetings(state):
                """Fetch meetings for a specific state"""
                try:
                    # Suppress 404 errors during parallel search
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    result = self.get_meetings(date, owning_authority_code=state)
                    sys.stdout = old_stdout
                    return result
                except:
                    sys.stdout = old_stdout
                    return []

            # Execute all state queries in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(fetch_state_meetings, state): state for state in states}

                for future in as_completed(futures):
                    meetings = future.result()
                    all_meetings.extend(meetings)

            # Find the meeting that matches the requested track code
            for m in all_meetings:
                if m.get('trackCode') == track_code:
                    meeting = m
                    break

        if not meeting:
            raise ValueError(f"No meeting found for {track_code} on {date}")

        meeting_id = meeting['meetingId']

        # Step 2: Get races for this meeting
        races = self.get_races(meeting_id)

        # Find the specific race
        target_race = None
        for race in races:
            if race.get('raceNumber') == race_number:
                target_race = race
                break

        if not target_race:
            # Debug: show available races
            available_races = [r.get('raceNumber') for r in races]
            raise ValueError(f"Race {race_number} not found at {track_code} on {date}. Available races: {available_races}")

        # The race object already contains all runner data in the 'runs' array
        # No need to call the /isolynx endpoint which requires additional permissions
        return {
            'meeting': meeting,
            'race': target_race
        }

    def get_bulk_runs_by_day(self, owning_authority_code: str, year: int, month: int, day: int) -> List[Dict[str, Any]]:
        """
        Get bulk runs data for a specific day (includes split times)

        Args:
            owning_authority_code: State code (e.g., 'VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'NZ')
            year: Year (e.g., 2024)
            month: Month (1-12)
            day: Day (1-31)

        Returns:
            List of run dictionaries with split time data
        """
        endpoint = f"/bulk/runs/{owning_authority_code}/{year}/{month}/{day}"
        return self._make_request(endpoint)

    def get_bulk_runs_by_month(self, owning_authority_code: str, year: int, month: int) -> List[Dict[str, Any]]:
        """
        Get bulk runs data for an entire month (includes split times)

        Args:
            owning_authority_code: State code (e.g., 'VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'NZ')
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            List of run dictionaries with split time data
        """
        endpoint = f"/bulk/runs/{owning_authority_code}/{year}/{month}"
        return self._make_request(endpoint)

    def get_tracks_for_date(self, date: str) -> List[Dict[str, str]]:
        """
        Get all tracks that have meetings on a specific date (across all Australian states and NZ)

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of dictionaries with trackCode and trackName
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Query all states in parallel to speed up loading
        states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']
        all_meetings = []

        def fetch_state_meetings(state):
            """Fetch meetings for a specific state"""
            try:
                meetings = self.get_meetings(date, owning_authority_code=state)
                return meetings
            except Exception as e:
                # If a state has no meetings or API returns 404, return empty list
                return []

        # Execute all state queries in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_state_meetings, state): state for state in states}

            for future in as_completed(futures):
                meetings = future.result()
                all_meetings.extend(meetings)

        # Get unique tracks
        tracks = {}
        for meeting in all_meetings:
            track_code = meeting.get('trackCode')
            track_name = meeting.get('trackName')
            if track_code and track_name:
                tracks[track_code] = track_name

        # Sort by track name
        return sorted([{'trackCode': code, 'trackName': name} for code, name in tracks.items()],
                     key=lambda x: x['trackName'])


if __name__ == "__main__":
    # Test the API
    API_KEY = "313c5027-4e3b-4f5b-a1b4-3608153dbaa3"

    api = TopazAPI(API_KEY)

    # Test 1: Get track codes
    print("Getting track codes...")
    tracks = api.get_track_codes()
    print(f"Found {len(tracks)} tracks")
    for track in tracks[:5]:
        print(f"  {track['trackCode']}: {track['trackName']}")

    # Test 2: Get meetings for a date
    print("\nGetting meetings for 2024-12-03...")
    meetings = api.get_meetings("2024-12-03")
    print(f"Found {len(meetings)} meetings")
    for meeting in meetings[:3]:
        print(f"  {meeting['trackName']} - {meeting['meetingType']}")

    # Test 3: Get tracks for a date
    print("\nGetting tracks for 2024-12-03...")
    tracks_for_date = api.get_tracks_for_date("2024-12-03")
    print(f"Found {len(tracks_for_date)} tracks with meetings")
    for track in tracks_for_date:
        print(f"  {track['trackCode']}: {track['trackName']}")
