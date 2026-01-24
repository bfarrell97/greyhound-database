"""
FAST Parallel Historical Data Populator - OPTIMIZED with BATCH INSERTS
Uses executemany and caching for 10-50x faster imports
Now includes extra fields: JumpCode, AverageSpeed, Sex, Breeding, Grade changes, etc.
Uses INSERT OR REPLACE to update existing entries with new data.

For 2020-2025 data: ~72 months x 6 states = 432 API calls
"""

import sqlite3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time
import sys
import threading

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY

class FastHistoricalPopulator:
    def __init__(self, db_path='greyhound_racing.db'):
        self.api = TopazAPI(TOPAZ_API_KEY)
        self.db_path = db_path
        self.runs_total = 0
        self.races_total = 0
        self.errors_total = 0
        self.greyhound_updates = {}  # Track greyhound breeding/sex updates
        
        # Add new columns if needed
        self._add_new_columns()
        
        # Pre-load caches for faster lookups
        self._init_caches()
    
    def _add_new_columns(self):
        """Add new columns if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entry_columns = [
            ("JumpCode", "TEXT"),
            ("AverageSpeed", "REAL"),
            ("FirstSplitPosition", "INTEGER"),
            ("SecondSplitTime", "REAL"),
            ("SecondSplitPosition", "INTEGER"),
            ("IncomingGrade", "TEXT"),
            ("OutgoingGrade", "TEXT"),
            ("GradedTo", "TEXT"),
            ("PrizeMoney", "REAL"),
            ("CareerPrizeMoney", "REAL"),
        ]
        
        greyhound_columns = [
            ("SireID", "INTEGER"),
            ("SireName", "TEXT"),
            ("DamID", "INTEGER"),
            ("DamName", "TEXT"),
            ("DateWhelped", "TEXT"),
            ("Sex", "TEXT"),
            ("ColourCode", "TEXT"),
        ]
        
        cursor.execute("PRAGMA table_info(GreyhoundEntries)")
        existing_entry_cols = {row[1] for row in cursor.fetchall()}
        
        for col_name, col_type in entry_columns:
            if col_name not in existing_entry_cols:
                print(f"  Adding GreyhoundEntries.{col_name}")
                cursor.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {col_name} {col_type}")
        
        cursor.execute("PRAGMA table_info(Greyhounds)")
        existing_greyhound_cols = {row[1] for row in cursor.fetchall()}
        
        for col_name, col_type in greyhound_columns:
            if col_name not in existing_greyhound_cols:
                print(f"  Adding Greyhounds.{col_name}")
                cursor.execute(f"ALTER TABLE Greyhounds ADD COLUMN {col_name} {col_type}")
        
        conn.commit()
        conn.close()
    
    def _init_caches(self):
        """Pre-load all existing IDs into memory for fast lookups"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cache tracks
        cursor.execute("SELECT TrackName, TrackID FROM Tracks")
        self.track_cache = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Cache greyhounds
        cursor.execute("SELECT GreyhoundName, GreyhoundID FROM Greyhounds")
        self.dog_cache = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Cache trainers
        cursor.execute("SELECT TrainerName, TrainerID FROM Trainers")
        self.trainer_cache = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Cache meetings (date_track -> ID)
        cursor.execute("""
            SELECT rm.MeetingDate || '_' || t.TrackName, rm.MeetingID 
            FROM RaceMeetings rm JOIN Tracks t ON rm.TrackID = t.TrackID
        """)
        self.meeting_cache = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Cache races (meetingid_racenum -> ID)
        cursor.execute("SELECT MeetingID || '_' || RaceNumber, RaceID FROM Races")
        self.race_cache = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Cache existing entries (raceid_dogid) - NOT USED anymore since we use REPLACE
        # cursor.execute("SELECT RaceID || '_' || GreyhoundID FROM GreyhoundEntries")
        # self.entry_cache = set(row[0] for row in cursor.fetchall())
        self.entry_cache = set()  # Empty - we'll replace all entries
        
        conn.close()
        print(f"  Caches loaded: {len(self.dog_cache):,} dogs, {len(self.track_cache)} tracks, "
              f"{len(self.race_cache):,} races (will REPLACE existing entries)")
        
    def fetch_month_data(self, state, year, month, max_retries=5):
        """Fetch data for one state/month with retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                runs = self.api.get_bulk_runs_by_month(state, year, month)
                return (state, year, month, runs, None)
            except Exception as e:
                error_str = str(e)
                # Check for rate limit (429) error
                if '429' in error_str or 'Too Many Requests' in error_str:
                    wait_time = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                    print(f"    Rate limited on {state}/{year}/{month}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return (state, year, month, [], error_str)
        return (state, year, month, [], "Max retries exceeded (rate limited)")
    
    def process_and_commit_month(self, state, year, month, runs):
        """Process runs for one month using BATCH inserts - much faster!"""
        if not runs:
            return 0, 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent performance
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        runs_added = 0
        races_added = 0
        
        try:
            # Collect all new entities to batch insert
            new_tracks = []
            new_dogs = []
            new_trainers = []
            new_meetings = []
            new_races = []
            new_entries = []
            
            # Group by meeting and race
            meetings = defaultdict(lambda: defaultdict(list))
            
            for run in runs:
                if run.get('resultTime') is None and not run.get('scratched'):
                    continue
                
                meeting_date = run['meetingDate'][:10]
                track_name = run['trackName']
                race_number = run['raceNumber']
                
                meeting_key = (meeting_date, track_name)
                meetings[meeting_key][race_number].append(run)
            
            # First pass: collect all new tracks, dogs, trainers
            for (meeting_date, track_name), races in meetings.items():
                # Track
                if track_name not in self.track_cache:
                    new_tracks.append((track_name,))
                
                for race_number, race_runs in races.items():
                    for run in race_runs:
                        if run.get('scratched'):
                            continue
                        
                        dog_name = run.get('dogName', '')
                        if dog_name and dog_name not in self.dog_cache:
                            new_dogs.append((dog_name, run.get('sireName', ''), run.get('damName', '')))
                        
                        trainer_name = run.get('trainerName', '')
                        if trainer_name and trainer_name not in self.trainer_cache:
                            new_trainers.append((trainer_name,))
            
            # Batch insert new tracks
            if new_tracks:
                cursor.executemany("INSERT OR IGNORE INTO Tracks (TrackName) VALUES (?)", new_tracks)
                cursor.execute("SELECT TrackName, TrackID FROM Tracks WHERE TrackName IN ({})".format(
                    ','.join('?' * len(new_tracks))), [t[0] for t in new_tracks])
                for row in cursor.fetchall():
                    self.track_cache[row[0]] = row[1]
            
            # Batch insert new dogs
            if new_dogs:
                cursor.executemany("INSERT OR IGNORE INTO Greyhounds (GreyhoundName, Sire, Dam) VALUES (?, ?, ?)", 
                                   new_dogs)
                cursor.execute("SELECT GreyhoundName, GreyhoundID FROM Greyhounds WHERE GreyhoundName IN ({})".format(
                    ','.join('?' * len(new_dogs))), [d[0] for d in new_dogs])
                for row in cursor.fetchall():
                    self.dog_cache[row[0]] = row[1]
            
            # Batch insert new trainers
            if new_trainers:
                unique_trainers = list(set(new_trainers))
                cursor.executemany("INSERT OR IGNORE INTO Trainers (TrainerName) VALUES (?)", unique_trainers)
                cursor.execute("SELECT TrainerName, TrainerID FROM Trainers WHERE TrainerName IN ({})".format(
                    ','.join('?' * len(unique_trainers))), [t[0] for t in unique_trainers])
                for row in cursor.fetchall():
                    self.trainer_cache[row[0]] = row[1]
            
            # Second pass: meetings and races
            for (meeting_date, track_name), races in meetings.items():
                track_id = self.track_cache.get(track_name)
                if not track_id:
                    continue
                
                meeting_key = f"{meeting_date}_{track_name}"
                if meeting_key not in self.meeting_cache:
                    new_meetings.append((track_id, meeting_date))
            
            # Batch insert meetings
            if new_meetings:
                cursor.executemany("INSERT OR IGNORE INTO RaceMeetings (TrackID, MeetingDate) VALUES (?, ?)", 
                                   new_meetings)
                # Refresh meeting cache
                cursor.execute("""
                    SELECT rm.MeetingDate || '_' || t.TrackName, rm.MeetingID 
                    FROM RaceMeetings rm JOIN Tracks t ON rm.TrackID = t.TrackID
                """)
                self.meeting_cache = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Collect races
            for (meeting_date, track_name), races in meetings.items():
                meeting_key = f"{meeting_date}_{track_name}"
                meeting_id = self.meeting_cache.get(meeting_key)
                if not meeting_id:
                    continue
                
                for race_number, race_runs in races.items():
                    race_key = f"{meeting_id}_{race_number}"
                    if race_key not in self.race_cache:
                        first_run = race_runs[0]
                        distance = first_run.get('distanceInMetres')
                        new_races.append((meeting_id, race_number, distance))
            
            # Batch insert races
            if new_races:
                cursor.executemany("INSERT OR IGNORE INTO Races (MeetingID, RaceNumber, Distance) VALUES (?, ?, ?)", 
                                   new_races)
                races_added = len(new_races)
                # Refresh race cache
                cursor.execute("SELECT MeetingID || '_' || RaceNumber, RaceID FROM Races")
                self.race_cache = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Third pass: entries
            for (meeting_date, track_name), races in meetings.items():
                meeting_key = f"{meeting_date}_{track_name}"
                meeting_id = self.meeting_cache.get(meeting_key)
                if not meeting_id:
                    continue
                
                for race_number, race_runs in races.items():
                    race_key = f"{meeting_id}_{race_number}"
                    race_id = self.race_cache.get(race_key)
                    if not race_id:
                        continue
                    
                    for run in race_runs:
                        if run.get('scratched'):
                            continue
                        
                        dog_name = run.get('dogName', '')
                        if not dog_name:
                            continue
                        
                        dog_id = self.dog_cache.get(dog_name)
                        if not dog_id:
                            continue
                        
                        trainer_name = run.get('trainerName', '')
                        trainer_id = self.trainer_cache.get(trainer_name)
                        
                        if run.get('unplaced'):
                            position = 'DNF'
                        else:
                            position = run.get('place')
                        
                        # Include all extra fields
                        new_entries.append((
                            race_id, dog_id, trainer_id,
                            run.get('boxNumber') or run.get('rugNumber'),
                            position,
                            run.get('resultTime'),
                            run.get('startPrice'),
                            run.get('weightInKg'),
                            run.get('firstSplitTime'),
                            run.get('pir'),  # InRun
                            run.get('last5'),  # Form
                            run.get('comment'),
                            run.get('rating'),
                            run.get('jumpCode'),
                            run.get('averageSpeed'),
                            run.get('firstSplitPosition'),
                            run.get('secondSplitTime'),
                            run.get('secondSplitPosition'),
                            run.get('incomingGrade'),
                            run.get('outgoingGrade'),
                            run.get('gradedTo'),
                            run.get('prizeMoney'),
                            run.get('careerPrizeMoney'),
                        ))
                        
                        # Also update greyhound with breeding/sex data
                        if dog_id not in self.greyhound_updates:
                            self.greyhound_updates[dog_id] = (
                                run.get('sireId'),
                                run.get('sireName'),
                                run.get('damId'),
                                run.get('damName'),
                                run.get('dateWhelped'),
                                run.get('sex'),
                                run.get('colourCode'),
                                dog_id,
                            )
            
            # Batch insert/replace entries with all fields
            if new_entries:
                cursor.executemany("""
                    INSERT OR REPLACE INTO GreyhoundEntries (
                        RaceID, GreyhoundID, TrainerID, Box, Position,
                        FinishTime, StartingPrice, Weight, Split,
                        InRun, Form, Comment, Rating,
                        JumpCode, AverageSpeed, FirstSplitPosition,
                        SecondSplitTime, SecondSplitPosition,
                        IncomingGrade, OutgoingGrade, GradedTo,
                        PrizeMoney, CareerPrizeMoney
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, new_entries)
                runs_added = len(new_entries)
            
            # Batch update greyhounds with breeding/sex data
            if self.greyhound_updates:
                cursor.executemany("""
                    UPDATE Greyhounds SET
                        SireID = COALESCE(?, SireID),
                        SireName = COALESCE(?, SireName),
                        DamID = COALESCE(?, DamID),
                        DamName = COALESCE(?, DamName),
                        DateWhelped = COALESCE(?, DateWhelped),
                        Sex = COALESCE(?, Sex),
                        ColourCode = COALESCE(?, ColourCode)
                    WHERE GreyhoundID = ?
                """, list(self.greyhound_updates.values()))
                self.greyhound_updates.clear()
            
            # COMMIT after this month!
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        return runs_added, races_added

    def populate_parallel(self, start_date, end_date, states=None, max_workers=3):
        """
        Populate data - fetch in parallel, but commit after each month
        Uses max 3 workers to avoid rate limiting
        """
        if states is None:
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS']
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate list of months
        months_to_process = []
        current = start
        while current <= end:
            if (current.year, current.month) not in months_to_process:
                months_to_process.append((current.year, current.month))
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        total_months = len(months_to_process)
        total_api_calls = total_months * len(states)
        
        print(f"{'='*80}")
        print(f"FAST PARALLEL IMPORT (with monthly commits)")
        print(f"{'='*80}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"States: {', '.join(states)}")
        print(f"Months: {total_months}")
        print(f"Total API calls: {total_api_calls}")
        print(f"Workers: {max_workers}")
        print(f"{'='*80}")
        print(f"\nProgress is saved after each month - safe to interrupt!")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Process month by month (commits after each month)
        for month_idx, (year, month) in enumerate(months_to_process):
            month_name = datetime(year, month, 1).strftime('%B %Y')
            month_start = time.time()
            
            print(f"\n[{month_idx+1}/{total_months}] {month_name}")
            print(f"  Fetching data for {len(states)} states in parallel...", end=" ", flush=True)
            
            # Fetch states for this month (max 3 at a time to avoid rate limits)
            month_data = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.fetch_month_data, state, year, month): state 
                          for state in states}
                
                for future in as_completed(futures):
                    result = future.result()
                    month_data.append(result)
                    time.sleep(1)  # Small delay between completions
            
            total_runs_fetched = sum(len(d[3]) for d in month_data)
            print(f"Got {total_runs_fetched:,} runs")
            
            # Process each state's data (sequential to avoid SQLite locks)
            month_runs = 0
            month_races = 0
            
            for state, y, m, runs, error in month_data:
                if error:
                    print(f"    {state}: ERROR - {error}")
                    self.errors_total += 1
                    continue
                
                if runs:
                    try:
                        runs_added, races_added = self.process_and_commit_month(state, y, m, runs)
                        month_runs += runs_added
                        month_races += races_added
                        print(f"    {state}: +{runs_added:,} runs, +{races_added:,} races (committed)")
                    except Exception as e:
                        print(f"    {state}: ERROR inserting - {e}")
                        self.errors_total += 1
                else:
                    print(f"    {state}: No data")
            
            self.runs_total += month_runs
            self.races_total += month_races
            
            elapsed = time.time() - start_time
            month_elapsed = time.time() - month_start
            
            # Rate limit prevention: wait 5 seconds between months
            if month_idx < len(months_to_process) - 1:
                print(f"  Waiting 5s to avoid rate limit...")
                time.sleep(5)
            remaining_months = total_months - (month_idx + 1)
            avg_per_month = elapsed / (month_idx + 1)
            eta = remaining_months * avg_per_month
            
            print(f"  Month complete: {month_runs:,} runs in {month_elapsed:.1f}s | "
                  f"Total: {self.runs_total:,} | ETA: {eta/60:.1f} min")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"IMPORT COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Races inserted: {self.races_total:,}")
        print(f"Runs inserted: {self.runs_total:,}")
        print(f"Errors: {self.errors_total}")
        if total_time > 0:
            print(f"Speed: {self.runs_total / total_time:.0f} runs/second")
        print(f"{'='*80}")


def main():
    print("="*80)
    print("FAST PARALLEL HISTORICAL DATA IMPORT (OPTIMIZED)")
    print("Uses batch inserts + caching for 10-50x speed")
    print("Commits after each month - safe to interrupt")
    print("="*80)
    
    # Get date range
    print("\nStart date (YYYY-MM-DD) [default: 2020-01-01]:")
    start_input = input("> ").strip()
    if not start_input:
        start_input = "2020-01-01"
    
    print("\nEnd date (YYYY-MM-DD) [default: 2022-12-31]:")
    end_input = input("> ").strip()
    if not end_input:
        end_input = "2022-12-31"
    
    # States
    print("\nStates (space-separated, or 'all') [default: all]:")
    print("Available: VIC NSW QLD SA WA TAS")
    state_input = input("> ").strip().upper()
    
    if state_input and state_input != 'ALL':
        states = [s for s in state_input.split() if s in ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS']]
        if not states:
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS']
    else:
        states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS']
    
    # Workers
    print("\nParallel workers [default: 3]:")
    worker_input = input().strip()
    workers = int(worker_input) if worker_input else 3
    
    # Confirm
    print(f"\n{'='*80}")
    print(f"Ready to import:")
    print(f"  Date range: {start_input} to {end_input}")
    print(f"  States: {', '.join(states)}")
    print(f"  Workers: {workers}")
    print(f"{'='*80}")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    print("\nInitializing (loading caches)...")
    populator = FastHistoricalPopulator()
    populator.populate_parallel(start_input, end_input, states, workers)


if __name__ == "__main__":
    main()
