"""
Greyhound Racing Database Module
Based on Hong Kong Racing Database structure
Stores race data, greyhound information, and sectional times
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json


class GreyhoundDatabase:
    """Database handler for Greyhound racing data"""

    def __init__(self, db_path='greyhound_racing.db'):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def get_connection(self):
        """Get or create database connection"""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def create_tables(self):
        """Create database tables for greyhound racing"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Tracks table (Australian and NZ greyhound tracks)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Tracks (
                TrackID INTEGER PRIMARY KEY AUTOINCREMENT,
                TrackKey TEXT UNIQUE NOT NULL,
                TrackName TEXT NOT NULL,
                State TEXT,
                Country TEXT
            )
        ''')

        # Greyhounds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Greyhounds (
                GreyhoundID INTEGER PRIMARY KEY AUTOINCREMENT,
                GreyhoundName TEXT NOT NULL,
                GreyhoundNameUnformatted TEXT,
                Sire TEXT,
                Dam TEXT,
                DateOfBirth TEXT,
                Color TEXT,
                Sex TEXT,
                Starts INTEGER DEFAULT 0,
                Wins INTEGER DEFAULT 0,
                Seconds INTEGER DEFAULT 0,
                Thirds INTEGER DEFAULT 0,
                Prizemoney REAL DEFAULT 0.0,
                WinPercentage REAL DEFAULT 0.0,
                PlacePercentage REAL DEFAULT 0.0,
                BestTime REAL,
                UNIQUE(GreyhoundName)
            )
        ''')

        # Trainers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Trainers (
                TrainerID INTEGER PRIMARY KEY AUTOINCREMENT,
                TrainerName TEXT UNIQUE NOT NULL,
                Location TEXT
            )
        ''')

        # Owners table (greyhounds have owners, not jockeys)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Owners (
                OwnerID INTEGER PRIMARY KEY AUTOINCREMENT,
                OwnerName TEXT UNIQUE NOT NULL
            )
        ''')

        # Race Meetings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS RaceMeetings (
                MeetingID INTEGER PRIMARY KEY AUTOINCREMENT,
                MeetingDate TEXT NOT NULL,
                TrackID INTEGER NOT NULL,
                FOREIGN KEY (TrackID) REFERENCES Tracks(TrackID)
            )
        ''')

        # Races table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Races (
                RaceID INTEGER PRIMARY KEY AUTOINCREMENT,
                MeetingID INTEGER NOT NULL,
                RaceNumber INTEGER NOT NULL,
                RaceName TEXT,
                Grade TEXT,
                Distance INTEGER,
                RaceTime TEXT,
                PrizeMoney TEXT,
                RaceCode TEXT,
                FOREIGN KEY (MeetingID) REFERENCES RaceMeetings(MeetingID)
            )
        ''')

        # Greyhound Entries table (greyhounds in specific races)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS GreyhoundEntries (
                EntryID INTEGER PRIMARY KEY AUTOINCREMENT,
                RaceID INTEGER NOT NULL,
                GreyhoundID INTEGER NOT NULL,
                Box INTEGER,
                Weight REAL,
                TrainerID INTEGER,
                OwnerID INTEGER,
                Position TEXT,
                Margin REAL,
                RunningPosition TEXT,
                FinishTime REAL,
                Split REAL,
                InRun TEXT,
                StartingPrice TEXT,
                Form TEXT,
                Comment TEXT,
                EarlySpeed INTEGER,
                Rating INTEGER,
                BSP REAL,
                BSPPlace REAL,
                FOREIGN KEY (RaceID) REFERENCES Races(RaceID),
                FOREIGN KEY (GreyhoundID) REFERENCES Greyhounds(GreyhoundID),
                FOREIGN KEY (TrainerID) REFERENCES Trainers(TrainerID),
                FOREIGN KEY (OwnerID) REFERENCES Owners(OwnerID)
            )
        ''')

        # Sectional Times table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS SectionalTimes (
                SectionalID INTEGER PRIMARY KEY AUTOINCREMENT,
                EntryID INTEGER NOT NULL,
                Sectional100m REAL,
                Sectional200m REAL,
                Sectional300m REAL,
                Sectional400m REAL,
                Sectional500m REAL,
                Sectional600m REAL,
                Sectional700m REAL,
                Sectional800m REAL,
                FinishTime REAL,
                Split REAL,
                FOREIGN KEY (EntryID) REFERENCES GreyhoundEntries(EntryID)
            )
        ''')

        # Adjustments table (race/greyhound/meeting adjustments)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Adjustments (
                AdjustmentID INTEGER PRIMARY KEY AUTOINCREMENT,
                EntryID INTEGER NOT NULL,
                RaceAdj REAL DEFAULT 0.0,
                GreyhoundAdj REAL DEFAULT 0.0,
                MeetingAdj REAL DEFAULT 0.0,
                WeightAdj REAL DEFAULT 0.0,
                GreyhoundMeetingAdj REAL DEFAULT 0.0,
                RaceTempo REAL DEFAULT 0.0,
                AdjustedTime REAL,
                FOREIGN KEY (EntryID) REFERENCES GreyhoundEntries(EntryID)
            )
        ''')

        # Benchmarks table (for track/distance average times)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Benchmarks (
                BenchmarkID INTEGER PRIMARY KEY AUTOINCREMENT,
                TrackName TEXT NOT NULL,
                Distance INTEGER NOT NULL,
                Grade TEXT,
                AvgTime REAL,
                MedianTime REAL,
                FastestTime REAL,
                SlowestTime REAL,
                StdDev REAL,
                SampleSize INTEGER,
                DateCreated TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(TrackName, Distance, Grade)
            )
        ''')

        # Add split time columns if they don't exist
        try:
            cursor.execute("ALTER TABLE Benchmarks ADD COLUMN AvgSplit REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE Benchmarks ADD COLUMN MedianSplit REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE Benchmarks ADD COLUMN FastestSplit REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE Benchmarks ADD COLUMN SplitSampleSize INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
        # Dog Feature Cache (Computed Features for ML)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS DogFeatureCache (
                DogID INTEGER,
                RaceID INTEGER, -- Features context (e.g. for a specific upcoming race)
                CalculationDate DATE DEFAULT CURRENT_DATE,
                FeaturesJSON TEXT,
                PRIMARY KEY (DogID, RaceID)
            )
        ''')  # Column already exists

        conn.commit()

    def add_or_get_track(self, track_key, track_name, state=None, country='AUS'):
        """Add track or return existing ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO Tracks (TrackKey, TrackName, State, Country)
            VALUES (?, ?, ?, ?)
        """, (track_key, track_name, state, country))

        cursor.execute("SELECT TrackID FROM Tracks WHERE TrackKey = ?", (track_key,))
        return cursor.fetchone()[0]

    def add_or_get_greyhound(self, greyhound_name, sire=None, dam=None,
                            starts=0, wins=0, prizemoney=0.0):
        """Add greyhound or return existing ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        win_pct = (wins / starts * 100) if starts > 0 else 0

        cursor.execute("""
            INSERT OR IGNORE INTO Greyhounds
            (GreyhoundName, GreyhoundNameUnformatted, Sire, Dam, Starts, Wins, Prizemoney, WinPercentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (greyhound_name, greyhound_name, sire, dam, starts, wins, prizemoney, win_pct))

        # Perform case-insensitive lookup to handle formatting differences (e.g. "SHIMA POLLY" vs "Shima Polly")
        # Order by GreyhoundID ASC to prefer the oldest record (containing history)
        cursor.execute("""
            SELECT GreyhoundID FROM Greyhounds 
            WHERE GreyhoundName = ? COLLATE NOCASE
            ORDER BY GreyhoundID ASC 
            LIMIT 1
        """, (greyhound_name,))
        
        result = cursor.fetchone()
        if result:
            return result[0]
            
        # Fallback (should be covered by INSERT above if new)
        cursor.execute("SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName = ?", (greyhound_name,))
        return cursor.fetchone()[0]

    def add_or_get_trainer(self, trainer_name, location=None):
        """Add trainer or return existing ID"""
        if not trainer_name:
            return None

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO Trainers (TrainerName, Location) VALUES (?, ?)",
                      (trainer_name, location))
        
        # Case-insensitive lookup, oldest first
        cursor.execute("""
            SELECT TrainerID FROM Trainers 
            WHERE TrainerName = ? COLLATE NOCASE
            ORDER BY TrainerID ASC
            LIMIT 1
        """, (trainer_name,))
        
        result = cursor.fetchone()
        return result[0] if result else None

    def add_or_get_owner(self, owner_name):
        """Add owner or return existing ID"""
        if not owner_name:
            return None

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("INSERT OR IGNORE INTO Owners (OwnerName) VALUES (?)", (owner_name,))
        
        # Case-insensitive lookup, oldest first
        cursor.execute("""
            SELECT OwnerID FROM Owners 
            WHERE OwnerName = ? COLLATE NOCASE
            ORDER BY OwnerID ASC
            LIMIT 1
        """, (owner_name,))
        
        result = cursor.fetchone()
        return result[0] if result else None

    def add_race_meeting(self, meeting_date, track_key):
        """Add race meeting if it doesn't exist, or return existing ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT TrackID FROM Tracks WHERE TrackKey = ?", (track_key,))
        track_id = cursor.fetchone()[0]

        # Check if meeting already exists for this date and track
        cursor.execute("""
            SELECT MeetingID FROM RaceMeetings
            WHERE MeetingDate = ? AND TrackID = ?
        """, (meeting_date, track_id))

        existing = cursor.fetchone()
        if existing:
            return existing[0]

        # Create new meeting
        cursor.execute("""
            INSERT INTO RaceMeetings (MeetingDate, TrackID)
            VALUES (?, ?)
        """, (meeting_date, track_id))

        return cursor.lastrowid

    def add_race(self, meeting_id, race_number, race_name='', grade='',
                 distance=0, race_time='', prize_money='', race_code=''):
        """Add race if it doesn't exist, or return existing ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if race already exists
        cursor.execute("""
            SELECT RaceID FROM Races
            WHERE MeetingID = ? AND RaceNumber = ?
        """, (meeting_id, race_number))

        existing = cursor.fetchone()
        if existing:
            # Update existing race info
            race_id = existing[0]
            cursor.execute("""
                UPDATE Races
                SET RaceName = ?, Grade = ?, Distance = ?, RaceTime = ?,
                    PrizeMoney = ?, RaceCode = ?
                WHERE RaceID = ?
            """, (race_name, grade, distance, race_time, prize_money, race_code, race_id))
            return race_id

        # Create new race
        cursor.execute("""
            INSERT INTO Races (MeetingID, RaceNumber, RaceName, Grade, Distance,
                              RaceTime, PrizeMoney, RaceCode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (meeting_id, race_number, race_name, grade, distance, race_time,
              prize_money, race_code))

        return cursor.lastrowid

    def add_entry(self, race_id, greyhound_id, box=None, weight=None,
                  trainer_id=None, owner_id=None, position=None, margin=None,
                  finish_time=None, split=None, starting_price=None, bsp=None, form=None):
        """
        Add greyhound entry to a race, or update if duplicate exists

        Duplicates are identified by: RaceID + GreyhoundID
        (i.e., same greyhound in same race)
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if entry already exists (same greyhound in same race)
        cursor.execute("""
            SELECT EntryID FROM GreyhoundEntries
            WHERE RaceID = ? AND GreyhoundID = ?
        """, (race_id, greyhound_id))

        existing = cursor.fetchone()
        if existing:
            # Update existing entry (overwrite duplicate)
            entry_id = existing[0]
            cursor.execute("""
                UPDATE GreyhoundEntries
                SET Box = ?, Weight = ?, TrainerID = ?, OwnerID = ?,
                    Position = ?, Margin = ?, FinishTime = ?, Split = ?,
                    StartingPrice = COALESCE(?, StartingPrice), 
                    BSP = COALESCE(?, BSP), 
                    InRun = ?
                WHERE EntryID = ?
            """, (box, weight, trainer_id, owner_id, position, margin,
                  finish_time, split, starting_price, bsp, form, entry_id))

            # Get greyhound name for logging
            cursor.execute("SELECT GreyhoundName FROM Greyhounds WHERE GreyhoundID = ?", (greyhound_id,))
            dog_name = cursor.fetchone()
            if dog_name:
                print(f"  [UPDATE] {dog_name[0]} (Box {box}, Pos {position})")

            return entry_id

        # Create new entry
        cursor.execute("""
            INSERT INTO GreyhoundEntries
            (RaceID, GreyhoundID, Box, Weight, TrainerID, OwnerID, Position,
             Margin, FinishTime, Split, StartingPrice, BSP, InRun)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (race_id, greyhound_id, box, weight, trainer_id, owner_id,
              position, margin, finish_time, split, starting_price, bsp, form))

        # Get greyhound name for logging
        cursor.execute("SELECT GreyhoundName FROM Greyhounds WHERE GreyhoundID = ?", (greyhound_id,))
        dog_name = cursor.fetchone()
        if dog_name:
            print(f"  [NEW] {dog_name[0]} (Box {box}, Pos {position})")

        return cursor.lastrowid

    def add_sectional_times(self, entry_id, sectionals_dict):
        """Add sectional times for an entry"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if sectionals already exist
        cursor.execute("SELECT SectionalID FROM SectionalTimes WHERE EntryID = ?", (entry_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing sectionals
            cursor.execute("""
                UPDATE SectionalTimes
                SET Sectional100m = ?, Sectional200m = ?, Sectional300m = ?,
                    Sectional400m = ?, Sectional500m = ?, Sectional600m = ?,
                    Sectional700m = ?, Sectional800m = ?, FinishTime = ?, Split = ?
                WHERE EntryID = ?
            """, (
                sectionals_dict.get('100m'), sectionals_dict.get('200m'),
                sectionals_dict.get('300m'), sectionals_dict.get('400m'),
                sectionals_dict.get('500m'), sectionals_dict.get('600m'),
                sectionals_dict.get('700m'), sectionals_dict.get('800m'),
                sectionals_dict.get('finish_time'), sectionals_dict.get('split'),
                entry_id
            ))
        else:
            # Insert new sectionals
            cursor.execute("""
                INSERT INTO SectionalTimes
                (EntryID, Sectional100m, Sectional200m, Sectional300m, Sectional400m,
                 Sectional500m, Sectional600m, Sectional700m, Sectional800m,
                 FinishTime, Split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                sectionals_dict.get('100m'), sectionals_dict.get('200m'),
                sectionals_dict.get('300m'), sectionals_dict.get('400m'),
                sectionals_dict.get('500m'), sectionals_dict.get('600m'),
                sectionals_dict.get('700m'), sectionals_dict.get('800m'),
                sectionals_dict.get('finish_time'), sectionals_dict.get('split')
            ))

        conn.commit()

    def get_greyhound_form(self, greyhound_name, limit=10):
        """Get recent form for a greyhound"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Build query with or without LIMIT based on whether limit is specified
        if limit is None:
            # Get all records without limit
            cursor.execute("""
                SELECT
                    rm.MeetingDate,
                    t.TrackName,
                    r.Distance,
                    r.Grade,
                    ge.Box as BoxNumber,
                    ge.Position,
                    ge.Margin,
                    ge.FinishTime,
                    ge.Split as FirstSectional,
                    COALESCE(ge.InRun, ge.Form) as RunningPosition,
                    ge.StartingPrice,
                    tr.TrainerName,
                    ge.SplitBenchmarkLengths as GFirstSecADJ,
                    rm.MeetingSplitAvgBenchmarkLengths as MFirstSecADJ,
                    ge.FinishTimeBenchmarkLengths as GOTADJ,
                    rm.MeetingAvgBenchmarkLengths as MOTADJ
                FROM GreyhoundEntries ge
                JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                LEFT JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
                WHERE g.GreyhoundName = ?
                ORDER BY rm.MeetingDate DESC, r.RaceNumber DESC
            """, (greyhound_name,))
        else:
            # Get limited records
            cursor.execute("""
                SELECT
                    rm.MeetingDate,
                    t.TrackName,
                    r.Distance,
                    r.Grade,
                    ge.Box as BoxNumber,
                    ge.Position,
                    ge.Margin,
                    ge.FinishTime,
                    ge.Split as FirstSectional,
                    COALESCE(ge.InRun, ge.Form) as RunningPosition,
                    ge.StartingPrice,
                    tr.TrainerName,
                    ge.SplitBenchmarkLengths as GFirstSecADJ,
                    rm.MeetingSplitAvgBenchmarkLengths as MFirstSecADJ,
                    ge.FinishTimeBenchmarkLengths as GOTADJ,
                    rm.MeetingAvgBenchmarkLengths as MOTADJ
                FROM GreyhoundEntries ge
                JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                LEFT JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
                WHERE g.GreyhoundName = ?
                ORDER BY rm.MeetingDate DESC, r.RaceNumber DESC
                LIMIT ?
            """, (greyhound_name, limit))

        return cursor.fetchall()

    def import_race_data(self, race_data):
        """Import race data from scraper"""
        conn = self.get_connection()

        try:
            # Add track
            track_id = self.add_or_get_track(
                race_data.get('track_key', ''),
                race_data.get('track_name', ''),
                race_data.get('state', ''),
                race_data.get('country', 'AUS')
            )

            # Add meeting
            meeting_id = self.add_race_meeting(
                race_data.get('date', ''),
                race_data.get('track_key', '')
            )

            # Add race
            race_id = self.add_race(
                meeting_id,
                race_data.get('race_number', 0),
                race_data.get('race_name', ''),
                race_data.get('grade', ''),
                race_data.get('distance', 0),
                race_data.get('race_time', ''),
                race_data.get('prize_money', ''),
                race_data.get('race_code', '')
            )

            # Add greyhounds and entries
            for entry in race_data.get('entries', []):
                greyhound_id = self.add_or_get_greyhound(
                    entry.get('greyhound_name', ''),
                    entry.get('sire', ''),
                    entry.get('dam', ''),
                    entry.get('starts', 0),
                    entry.get('wins', 0),
                    entry.get('prizemoney', 0.0)
                )

                trainer_id = self.add_or_get_trainer(entry.get('trainer', ''))
                owner_id = self.add_or_get_owner(entry.get('owner', ''))

                entry_id = self.add_entry(
                    race_id, greyhound_id,
                    box=entry.get('box'),
                    weight=entry.get('weight'),
                    trainer_id=trainer_id,
                    owner_id=owner_id,
                    position=entry.get('position'),
                    margin=entry.get('margin'),
                    finish_time=entry.get('finish_time'),
                    split=entry.get('split'),
                    starting_price=entry.get('starting_price'),
                    form=entry.get('form')
                )

                # Add sectional times if available
                if 'sectionals' in entry:
                    self.add_sectional_times(entry_id, entry['sectionals'])

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            print(f"Error importing race data: {e}")
            return False

    def import_results_data(self, results_data, meeting_date, track_name):
        """
        Import results data from scraper

        Automatically handles duplicates by overwriting existing records:
        - If same date + track + race number exists, updates that race
        - If same greyhound in that race exists, updates that entry

        Args:
            results_data: Dictionary from scrape_results() method
            meeting_date: Date string for the race meeting (YYYY-MM-DD)
            track_name: Track name
        """
        conn = self.get_connection()

        try:
            # Create track if it doesn't exist
            track_key = track_name.lower().replace(' ', '-')
            track_id = self.add_or_get_track(track_key, track_name)

            # Create meeting (or get existing)
            meeting_id = self.add_race_meeting(meeting_date, track_key)

            # Create race (or update if exists for same meeting + race number)
            race_id = self.add_race(
                meeting_id,
                results_data.get('race_number', 1),
                race_name=results_data.get('race_name', ''),
                grade=results_data.get('grade', ''),
                distance=results_data.get('distance', 0),
                race_time=results_data.get('race_time', ''),
                prize_money=results_data.get('prize_money', '')
            )

            print(f"Processing Race ID {race_id}: {track_name} Race {results_data.get('race_number', 1)} on {meeting_date}")

            # Import each greyhound entry
            for result in results_data.get('results', []):
                # Add greyhound
                greyhound_id = self.add_or_get_greyhound(
                    result.get('greyhound_name', ''),
                    sire=result.get('sire', ''),
                    dam=result.get('dam', '')
                )

                # Add trainer
                trainer_id = self.add_or_get_trainer(result.get('trainer', ''))

                # Add entry with results
                entry_id = self.add_entry(
                    race_id, greyhound_id,
                    box=result.get('box'),
                    weight=result.get('weight'),
                    trainer_id=trainer_id,
                    position=result.get('position'),
                    margin=result.get('margin'),
                    finish_time=result.get('finish_time'),
                    split=result.get('split'),
                    starting_price=result.get('starting_price'),
                    bsp=result.get('bsp'),
                    form=result.get('in_run')
                )

            # Fix winning margins (winner should show margin TO 2nd place, not FROM winner)
            self.fix_winning_margins(race_id)

            # Update all greyhound stats from their race entries
            self.update_greyhound_stats()

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            print(f"Error importing results data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def import_form_guide_data(self, form_data, meeting_date, track_name):
        """
        Import form guide data from scraper (upcoming race)

        Args:
            form_data: Dictionary from scrape_form_guide() method
            meeting_date: Date string for the race meeting (YYYY-MM-DD)
            track_name: Track name
        """
        conn = self.get_connection()

        try:
            # Create track if it doesn't exist
            track_key = track_name.lower().replace(' ', '-')
            track_id = self.add_or_get_track(track_key, track_name)

            # Create meeting
            meeting_id = self.add_race_meeting(meeting_date, track_key)

            # Create race
            race_id = self.add_race(
                meeting_id,
                form_data.get('race_number', 1),
                race_name=form_data.get('race_name', ''),
                grade=form_data.get('grade', ''),
                distance=form_data.get('distance', 0),
                race_time=form_data.get('race_time', ''),
                prize_money=form_data.get('prize_money', '')
            )

            # Import each greyhound entry
            for entry in form_data.get('entries', []):
                # Add greyhound with stats
                greyhound_id = self.add_or_get_greyhound(
                    entry.get('greyhound_name', ''),
                    sire=entry.get('sire', ''),
                    dam=entry.get('dam', ''),
                    starts=entry.get('starts', 0),
                    wins=entry.get('wins', 0),
                    prizemoney=entry.get('prizemoney', 0)
                )

                # Add trainer
                trainer_id = self.add_or_get_trainer(entry.get('trainer', ''))

                # Add owner if present
                owner_id = None
                if entry.get('owner'):
                    owner_id = self.add_or_get_owner(entry.get('owner'))

                # Add entry (upcoming race, no results yet)
                entry_id = self.add_entry(
                    race_id, greyhound_id,
                    box=entry.get('box'),
                    weight=entry.get('weight'),
                    trainer_id=trainer_id,
                    owner_id=owner_id,
                    starting_price=entry.get('starting_price') # Pass injected odds
                )

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            print(f"Error importing form guide data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_greyhound_stats(self, greyhound_id=None):
        """
        Update greyhound career statistics from race entries

        Args:
            greyhound_id: Specific greyhound to update, or None for all greyhounds
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if greyhound_id:
            where_clause = "WHERE ge.GreyhoundID = ?"
            params = (greyhound_id,)
        else:
            where_clause = ""
            params = ()

        # Calculate stats for each greyhound from their race entries
        # Note: Position is TEXT to allow 'DNF', so we use CAST for numeric comparisons
        query = f"""
            SELECT
                ge.GreyhoundID,
                COUNT(*) as total_starts,
                SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as total_wins,
                SUM(CASE WHEN ge.Position = '2' THEN 1 ELSE 0 END) as total_seconds,
                SUM(CASE WHEN ge.Position = '3' THEN 1 ELSE 0 END) as total_thirds,
                MIN(CASE WHEN ge.Position != 'DNF' THEN ge.FinishTime END) as best_time
            FROM GreyhoundEntries ge
            {where_clause}
            GROUP BY ge.GreyhoundID
        """

        cursor.execute(query, params)
        stats = cursor.fetchall()

        # Update each greyhound's stats
        for row in stats:
            gid, starts, wins, seconds, thirds, best_time = row

            # Calculate percentages
            win_pct = (wins / starts * 100) if starts > 0 else 0
            place_pct = ((wins + seconds + thirds) / starts * 100) if starts > 0 else 0

            cursor.execute("""
                UPDATE Greyhounds
                SET Starts = ?,
                    Wins = ?,
                    Seconds = ?,
                    Thirds = ?,
                    WinPercentage = ?,
                    PlacePercentage = ?,
                    BestTime = ?
                WHERE GreyhoundID = ?
            """, (starts, wins, seconds, thirds, win_pct, place_pct, best_time, gid))

        conn.commit()
        return len(stats)

    def fix_winning_margins(self, race_id):
        """
        Fix winning margins for a race

        The winner (position 1) should show the margin TO second place,
        not margin FROM winner (which is 0.00)

        Args:
            race_id: RaceID to fix margins for
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get the margin of the 2nd place dog
        cursor.execute("""
            SELECT Margin
            FROM GreyhoundEntries
            WHERE RaceID = ? AND Position = '2'
        """, (race_id,))

        result = cursor.fetchone()
        if result and result[0] is not None:
            second_place_margin = result[0]

            # Update the winner's margin to be the same as 2nd place
            cursor.execute("""
                UPDATE GreyhoundEntries
                SET Margin = ?
                WHERE RaceID = ? AND Position = '1'
            """, (second_place_margin, race_id))

            conn.commit()

    def remove_duplicate_entries(self):
        """
        Remove duplicate greyhound entries in races

        Keeps the most recently inserted entry for each (RaceID, GreyhoundID) combination
        and deletes older duplicates.

        Returns: Number of duplicate entries removed
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Find duplicates: multiple entries for same greyhound in same race
        cursor.execute("""
            SELECT RaceID, GreyhoundID, COUNT(*) as count,
                   GROUP_CONCAT(EntryID) as entry_ids
            FROM GreyhoundEntries
            GROUP BY RaceID, GreyhoundID
            HAVING COUNT(*) > 1
        """)

        duplicates = cursor.fetchall()
        total_removed = 0

        for race_id, greyhound_id, count, entry_ids_str in duplicates:
            entry_ids = [int(x) for x in entry_ids_str.split(',')]

            # Keep the highest EntryID (most recent), delete others
            keep_id = max(entry_ids)
            delete_ids = [eid for eid in entry_ids if eid != keep_id]

            # Get greyhound name for logging
            cursor.execute("SELECT GreyhoundName FROM Greyhounds WHERE GreyhoundID = ?", (greyhound_id,))
            result = cursor.fetchone()
            dog_name = result[0] if result else f"ID {greyhound_id}"

            print(f"Removing {len(delete_ids)} duplicate(s) for {dog_name} in Race {race_id}, keeping Entry {keep_id}")

            # Delete sectional times for duplicate entries
            for eid in delete_ids:
                cursor.execute("DELETE FROM SectionalTimes WHERE EntryID = ?", (eid,))
                cursor.execute("DELETE FROM Adjustments WHERE EntryID = ?", (eid,))

            # Delete duplicate entries
            placeholders = ','.join('?' * len(delete_ids))
            cursor.execute(f"DELETE FROM GreyhoundEntries WHERE EntryID IN ({placeholders})", delete_ids)

            total_removed += len(delete_ids)

        conn.commit()
        print(f"Total duplicates removed: {total_removed}")
        return total_removed

    def cleanup_stale_races(self, days_old=2):
        """
        Delete races older than X days that have NO results (incomplete/form guide data).
        
        Args:
            days_old: Number of days in the past to consider 'stale'
        Returns:
            Number of races deleted
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
            print(f"[CLEANUP] Deleting incomplete races before {cutoff_date}...")
            
            # Find stale races:
            # 1. Meeting date < cutoff
            # 2. Entries have NULL/Zero FinishTime (indicating no result)
            
            # Identify RACE IDs to delete
            # We look for races where ALL entries are incomplete
            cursor.execute("""
                SELECT r.RaceID, rm.MeetingDate, t.TrackName, r.RaceNumber
                FROM Races r
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                WHERE rm.MeetingDate < ?
                AND NOT EXISTS (
                    SELECT 1 FROM GreyhoundEntries ge 
                    WHERE ge.RaceID = r.RaceID 
                    AND (ge.FinishTime > 0 OR ge.Position IS NOT NULL AND ge.Position != 'DNF')
                )
            """, (cutoff_date,))
            
            stale_races = cursor.fetchall()
            
            if not stale_races:
                print("[CLEANUP] No stale races found.")
                return 0
                
            race_ids = [row[0] for row in stale_races]
            print(f"[CLEANUP] Found {len(race_ids)} stale races. Example: {stale_races[0][1]} {stale_races[0][2]} R{stale_races[0][3]}")
            
            # Delete logic (Child tables first)
            placeholders = ','.join('?' * len(race_ids))
            
            # 1. Entries
            cursor.execute(f"DELETE FROM GreyhoundEntries WHERE RaceID IN ({placeholders})", race_ids)
            entries_deleted = cursor.rowcount
            
            # 2. Races
            cursor.execute(f"DELETE FROM Races WHERE RaceID IN ({placeholders})", race_ids)
            races_deleted = cursor.rowcount
            
            # 3. Clean up empty meetings? (Optional, skipping for safety)
            
            conn.commit()
            print(f"[CLEANUP] Success. Deleted {races_deleted} races and {entries_deleted} incomplete entries.")
            return races_deleted
            
        except Exception as e:
            conn.rollback()
            print(f"[CLEANUP] Error: {e}")
            return 0

    def get_latest_result_date(self):
        """
        Get the date of the latest race with results in the database
        """
        try:
            cursor = self.get_connection().cursor()
            # Join with Races and Entries to ensure we have actual results
            query = """
                SELECT MAX(rm.MeetingDate) 
                FROM RaceMeetings rm
                JOIN Races r ON rm.MeetingID = r.MeetingID
                JOIN GreyhoundEntries ge ON r.RaceID = ge.RaceID
                WHERE ge.FinishTime > 0
            """
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result and result[0] else "None"
        except Exception as e:
            print(f"Error getting latest date: {e}")
            return "Error"

    def commit(self):
        """Commit changes"""
        if self.conn:
            self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
