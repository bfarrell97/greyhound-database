"""
Debug Merge Logic
Goal: Find why the merge between test_df and second_place_times is failing.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def debug_merge():
    print("Loading Tiny Sample Data...")
    conn = sqlite3.connect(DB_PATH)
    # Get 1 Safe Track, 1 Month
    query = """
    SELECT
        ge.GreyhoundID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE t.TrackName = 'Warrnambool' 
      AND rm.MeetingDate >= '2025-01-01'
      AND rm.MeetingDate < '2025-02-01'
      AND ge.FinishTime IS NOT NULL
    LIMIT 100
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    
    print(f"Loaded {len(df)} rows.")
    
    # Simulate feature engineering output
    df['PredOverall'] = df['FinishTime'] # Just use actual for debug
    
    # Create RaceKey exactly as in script
    df['RaceKey'] = df['MeetingDate'].astype(str) + df['TrackName'] + df['RaceID'].astype(str)
    
    print("\nSample RaceKeys:")
    print(df['RaceKey'].head().to_list())
    
    # Group logic
    sorted_preds = df[['RaceKey', 'PredOverall']].sort_values(['RaceKey', 'PredOverall'])
    
    # Method 1 used in script
    second_place_times = sorted_preds.groupby('RaceKey')['PredOverall'].nth(1).reset_index()
    second_place_times.columns = ['RaceKey', 'Time2nd']
    
    print("\nSecond Place Times (Sample):")
    print(second_place_times.head())
    
    # Check dtypes
    print("\nDTYPES:")
    print(f"Main DF RaceKey: {df['RaceKey'].dtype}")
    print(f"Lookup RaceKey: {second_place_times['RaceKey'].dtype}")
    
    # Attempt Merge
    merged = df.merge(second_place_times, on='RaceKey', how='left')
    
    print("\nMerge Result:")
    print(f"Original Rows: {len(df)}")
    print(f"Merged Rows: {len(merged)}")
    print(f"Non-Null Time2nd: {merged['Time2nd'].notna().sum()}")
    
    if merged['Time2nd'].notna().sum() == 0:
        print("[FAIL] Merge produced no matches.")
    else:
        print("[SUCCESS] Merge worked.")

if __name__ == "__main__":
    debug_merge()
