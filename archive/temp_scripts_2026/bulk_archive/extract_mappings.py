
import sqlite3
import pandas as pd
import json
import os

DB_PATH = 'greyhound_racing.db'
OUTPUT_MAP = 'models/v33_mappings.json'

def extract_mappings():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Load all unique values for categorical columns from the training range
    query = """
    SELECT 
        ge.Box, r.Distance, r.Grade, t.TrackName as Track
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2020-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    mappings = {}
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    
    for col in categorical_cols:
        # Fill NaNs with a placeholder to avoid sorting errors
        df[col] = df[col].fillna('N/A')
        # Get unique values and sort them to ensure deterministic encoding
        unique_vals = sorted([str(val) for val in df[col].unique().tolist()])
        mappings[col] = {val: i for i, val in enumerate(unique_vals)}
        print(f"Extracted {len(unique_vals)} categories for {col}")

    # Save to JSON
    os.makedirs('models', exist_ok=True)
    with open(OUTPUT_MAP, 'w') as f:
        json.dump(mappings, f, indent=4)
    
    print(f"\nSaved mapping to {OUTPUT_MAP}")

if __name__ == "__main__":
    extract_mappings()
