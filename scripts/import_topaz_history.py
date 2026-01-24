import sqlite3
import pandas as pd
import sys
import os
import time
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.getcwd())
try:
    from src.integration.topaz_api import TopazAPI
    from src.core.config import TOPAZ_API_KEY
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from integration.topaz_api import TopazAPI
    TOPAZ_API_KEY = "313c5027-4e3b-4f5b-a1b4-3608153dbaa3"

DB_PATH = 'greyhound_racing.db'
STATES = ['VIC', 'NSW', 'QLD', 'SA', 'WA', 'TAS', 'NZ']

def get_existing_entries_for_month(conn, year, month):
    """Load existing entries for matching to minimize DB hits."""
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
        
    query = f"""
    SELECT 
        ge.EntryID, ge.GreyhoundID, 
        r.RaceNumber, r.RaceTime,
        rm.MeetingDate, t.TrackName,
        g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '{start_date}' AND rm.MeetingDate < '{end_date}'
    """
    print(f"Loading local DB entries for {year}-{month:02d}...")
    df = pd.read_sql_query(query, conn)
    
    # Create a lookup key: Match on specific fields
    # Note: Track matching is hard. Topaz uses codes (MEA), we have codes in Track table? 
    # Let's hope TrackCode is populated in DB.
    
    # Simplify dog name for matching (remove spaces, upper)
    df['MatchName'] = df['GreyhoundName'].str.replace(' ', '').str.upper().str.replace(r'[^A-Z]', '', regex=True)
    df['MatchDate'] = pd.to_datetime(df['MeetingDate']).dt.strftime('%Y-%m-%d')
    
    return df

def update_db_batch(conn, updates):
    """Execute batch update."""
    if not updates: return
    cursor = conn.cursor()
    # Corrected Schema Mapping per User Request:
    # Split -> 'Split' column
    # firstsplitposition -> 'FirstSplitPosition' column
    # secondSplitTime -> 'SecondSplitTime' column
    
    cursor.executemany("""
        UPDATE GreyhoundEntries 
        SET Split = ?, SecondSplitTime = ?, FirstSplitPosition = ?, TopazComment = ?
        WHERE EntryID = ?
    """, updates)
    conn.commit()
    print(f"Updated {len(updates)} records.")

def import_month(year, month):
    api = TopazAPI(TOPAZ_API_KEY)
    conn = sqlite3.connect(DB_PATH)
    
    # Load Local Data
    local_df = get_existing_entries_for_month(conn, year, month)
    if local_df.empty:
        print("No local data for this month. Skipping.")
        conn.close()
        return

    print(f"Local entries loaded: {len(local_df)}")
    
    # State Loop
    updates = []
    
    for state in STATES:
        print(f"Fetching {state} data for {year}-{month:02d}...")
        try:
            runs = api.get_bulk_runs_by_month(state, year, month)
        except Exception as e:
            print(f"Failed to fetch {state}: {e}")
            continue
            
        if not runs:
            print(f"No runs for {state}")
            continue
            
        print(f"  Found {len(runs)} Topaz runs. Matching...")
        
        # Determine matches
        # We can iterate runs and lookup in local_df
        # Indexing local_df for speed
        # Key: Date + RaceNum + DogName (Normalized)
        # Note: Track is tricky if codes mismatch. Date+Race+Dog is usually unique enough globally (rarely same dog runs twice same day)
        
        # Build Lookup Dict
        # {(Date, RaceNum, DogName): EntryID}
        lookup = {}
        for _, row in local_df.iterrows():
            key = (row['MatchDate'], row['RaceNumber'], row['MatchName'])
            lookup[key] = row['EntryID']
            
        matched_count = 0
        
        for run in runs:
            # Extract Keys
            r_date = run.get('meetingDate', '').split('T')[0]
            r_num = run.get('raceNumber')
            r_dog = run.get('dogName', '').replace(' ', '').upper().replace("'", "").replace(".", "")
            
            key = (r_date, r_num, r_dog)
            
            if key in lookup:
                entry_id = lookup[key]
                
                # Extract Data
                split1 = run.get('split')
                split2 = run.get('secondSplitTime') # User didn't specify this one, keeping for now
                pir = run.get('firstsplitposition')
                comment = run.get('comment')
                
                updates.append((split1, split2, pir, comment, entry_id))
                matched_count += 1
                
        print(f"  Matched {matched_count} / {len(runs)} runs.")

    # Execute Updates
    if updates:
        print(f"Committing {len(updates)} updates to DB...")
        update_db_batch(conn, updates)
    else:
        print("No updates to commit.")
        
    conn.close()

if __name__ == "__main__":
    # Import recent history
    # Import recent history
    # Targeting CURRENT MONTH (Dec 2025)
    import_month(2025, 12)
