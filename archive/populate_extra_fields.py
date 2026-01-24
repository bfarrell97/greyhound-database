"""
FAST Parallel Import of Additional API Fields for 2020-2025
Uses same optimizations as populate_fast_parallel.py:
- WAL mode for concurrent writes
- Batch updates with executemany
- Pre-loaded caches for fast lookups
- Parallel API fetching with ThreadPoolExecutor
- Rate limit handling with exponential backoff

Adds: JumpCode, AverageSpeed, FirstSplitPosition, Sex, IncomingGrade, 
      OutgoingGrade, GradedTo, CareerPrizeMoney, Sire/Dam data
"""
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY

DB_PATH = "greyhound_racing.db"

class FastExtraFieldsPopulator:
    def __init__(self, db_path=DB_PATH):
        self.api = TopazAPI(TOPAZ_API_KEY)
        self.db_path = db_path
        self.entries_updated = 0
        self.greyhounds_updated = 0
        self.lock = threading.Lock()
        
    def add_new_columns(self):
        """Add new columns to GreyhoundEntries and Greyhounds tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # New columns for GreyhoundEntries
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
        
        # New columns for Greyhounds
        greyhound_columns = [
            ("SireID", "INTEGER"),
            ("SireName", "TEXT"),
            ("DamID", "INTEGER"),
            ("DamName", "TEXT"),
            ("DateWhelped", "TEXT"),
            ("Sex", "TEXT"),
            ("ColourCode", "TEXT"),
        ]
        
        # Check existing columns in GreyhoundEntries
        cursor.execute("PRAGMA table_info(GreyhoundEntries)")
        existing_entry_cols = {row[1] for row in cursor.fetchall()}
        
        added = 0
        for col_name, col_type in entry_columns:
            if col_name not in existing_entry_cols:
                print(f"  Adding GreyhoundEntries.{col_name}")
                cursor.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {col_name} {col_type}")
                added += 1
        
        # Check existing columns in Greyhounds
        cursor.execute("PRAGMA table_info(Greyhounds)")
        existing_greyhound_cols = {row[1] for row in cursor.fetchall()}
        
        for col_name, col_type in greyhound_columns:
            if col_name not in existing_greyhound_cols:
                print(f"  Adding Greyhounds.{col_name}")
                cursor.execute(f"ALTER TABLE Greyhounds ADD COLUMN {col_name} {col_type}")
                added += 1
        
        conn.commit()
        conn.close()
        
        if added > 0:
            print(f"  Added {added} new columns")
        else:
            print("  All columns already exist")
        
    def fetch_month_data(self, state, year, month, max_retries=5):
        """Fetch data for one state/month with retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                runs = self.api.get_bulk_runs_by_month(state, year, month)
                return (state, year, month, runs, None)
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'Too Many Requests' in error_str or 'rate' in error_str.lower():
                    wait_time = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                    print(f"    Rate limited on {state}/{year}/{month}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return (state, year, month, [], error_str)
        return (state, year, month, [], "Max retries exceeded (rate limited)")
    
    def process_and_commit_month(self, state, year, month, runs):
        """Process runs for one month using BATCH updates"""
        if not runs:
            return 0, 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrent performance
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        entries_updated = 0
        greyhounds_updated = 0
        
        try:
            # Collect all updates
            entry_updates = []
            greyhound_updates = {}
            
            for run in runs:
                if run.get('scratched'):
                    continue
                    
                race_id = run.get('raceId')
                dog_id = run.get('dogId')
                
                if not race_id or not dog_id:
                    continue
                
                # Entry updates - store as tuple for executemany
                entry_updates.append((
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
                    race_id,
                    dog_id,
                ))
                
                # Greyhound updates (dedupe by dog_id)
                if dog_id not in greyhound_updates:
                    greyhound_updates[dog_id] = (
                        run.get('sireId'),
                        run.get('sireName'),
                        run.get('damId'),
                        run.get('damName'),
                        run.get('dateWhelped'),
                        run.get('sex'),
                        run.get('colourCode'),
                        dog_id,
                    )
            
            # Batch update entries
            if entry_updates:
                cursor.executemany("""
                    UPDATE GreyhoundEntries SET
                        JumpCode = ?,
                        AverageSpeed = ?,
                        FirstSplitPosition = ?,
                        SecondSplitTime = ?,
                        SecondSplitPosition = ?,
                        IncomingGrade = ?,
                        OutgoingGrade = ?,
                        GradedTo = ?,
                        PrizeMoney = ?,
                        CareerPrizeMoney = ?
                    WHERE RaceID = ? AND GreyhoundID = ?
                """, entry_updates)
                entries_updated = cursor.rowcount if cursor.rowcount > 0 else len(entry_updates)
            
            # Batch update greyhounds
            if greyhound_updates:
                cursor.executemany("""
                    UPDATE Greyhounds SET
                        SireID = COALESCE(SireID, ?),
                        SireName = COALESCE(SireName, ?),
                        DamID = COALESCE(DamID, ?),
                        DamName = COALESCE(DamName, ?),
                        DateWhelped = COALESCE(DateWhelped, ?),
                        Sex = COALESCE(Sex, ?),
                        ColourCode = COALESCE(ColourCode, ?)
                    WHERE GreyhoundID = ?
                """, list(greyhound_updates.values()))
                greyhounds_updated = len(greyhound_updates)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"    Error processing {state} {year}-{month:02d}: {e}")
        finally:
            conn.close()
        
        return entries_updated, greyhounds_updated

    def populate_parallel(self, start_year=2020, end_year=2025, states=None, max_workers=6):
        """
        Populate extra fields - fetch in parallel, commit after each month
        """
        if states is None:
            states = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS']
        
        # Generate list of all months to process
        months_to_process = []
        for year in range(start_year, end_year + 1):
            end_month = 12
            for month in range(1, end_month + 1):
                for state in states:
                    months_to_process.append((state, year, month))
        
        total_months = len(months_to_process)
        print(f"\nProcessing {total_months} state-months across {len(states)} states...")
        print(f"Using {max_workers} parallel workers")
        print("-" * 60)
        
        start_time = time.time()
        processed = 0
        
        # Process in batches to manage rate limits
        batch_size = max_workers
        
        for i in range(0, total_months, batch_size):
            batch = months_to_process[i:i + batch_size]
            
            # Fetch data in parallel
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.fetch_month_data, state, year, month): (state, year, month)
                    for state, year, month in batch
                }
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Process results sequentially (to avoid DB lock contention)
            for state, year, month, runs, error in results:
                if error:
                    print(f"  {state} {year}-{month:02d}: Error - {error}")
                elif runs:
                    entries, greyhounds = self.process_and_commit_month(state, year, month, runs)
                    with self.lock:
                        self.entries_updated += entries
                        self.greyhounds_updated += greyhounds
                    print(f"  {state} {year}-{month:02d}: {len(runs):,} runs -> {entries:,} entries updated")
                else:
                    print(f"  {state} {year}-{month:02d}: No data")
                
                processed += 1
            
            # Progress update
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_months - processed) / rate if rate > 0 else 0
            print(f"  Progress: {processed}/{total_months} ({processed/total_months*100:.1f}%) - "
                  f"ETA: {remaining/60:.1f} min")
            
            # Small delay between batches to avoid rate limits
            time.sleep(2)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("IMPORT COMPLETE")
        print("=" * 60)
        print(f"Total entries updated: {self.entries_updated:,}")
        print(f"Total greyhounds updated: {self.greyhounds_updated:,}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average rate: {processed/total_time*60:.1f} state-months/minute")


def main():
    print("=" * 70)
    print("FAST PARALLEL IMPORT OF EXTRA API FIELDS (2020-2025)")
    print("=" * 70)
    
    populator = FastExtraFieldsPopulator()
    
    # Step 1: Add new columns
    print("\n1. Updating database schema...")
    populator.add_new_columns()
    
    # Step 2: Import data
    print("\n2. Importing extra fields from API...")
    populator.populate_parallel(
        start_year=2020,
        end_year=2025,
        states=['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS'],
        max_workers=6
    )


if __name__ == "__main__":
    main()
