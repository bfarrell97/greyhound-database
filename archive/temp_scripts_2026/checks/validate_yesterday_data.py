
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def validate_yesterday_data():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Calculate yesterday's date
    # yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    yesterday = "2025-12-18" # Hardcode test 2025-12-18 for duplicate check
    
    print(f"Validating data for Date: {yesterday}")
    
    # Get all races for yesterday
    query = f"""
    SELECT 
        r.RaceID, r.RaceNumber, t.TrackName, g.GreyhoundName, 
        ge.Box, ge.Position as Place, ge.FinishTime, ge.Split, ge.Weight, ge.StartingPrice, ge.BSP
    FROM Races r
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN GreyhoundEntries ge ON r.RaceID = ge.RaceID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = '{yesterday}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"CRITICAL: No data found for {yesterday}!")
        return

    print(f"Total Entries Found: {len(df)}")
    
    print("\n[INFO] All Tracks found for this date:")
    print(df['TrackName'].unique())

    # Columns to validate (Critical columns that should be full)
    # Note: 'Place' might be None for DNFs/Scratched? Scraper usually handles this.
    # 'Split' can be null sometimes if not recorded.
    
    critical_columns = ['Box', 'FinishTime', 'Weight', 'StartingPrice']
    
    issues = []
    
    for col in critical_columns:
        # Check for Nulls
        null_count = df[col].isnull().sum()
        
        # Check for empty strings or 0 strings if applicable
        empty_count = 0
        if df[col].dtype == object:
             empty_count = df[col].apply(lambda x: str(x).strip() == '' or str(x) == 'None').sum()
        
        if null_count > 0 or empty_count > 0:
            issues.append(f"Column '{col}' has {null_count} Nulls and {empty_count} Empty values.")
            
    # Check FinishTime > 0
    # Convert FinishTime to numeric, coerce errors
    df['FinishTime_Num'] = pd.to_numeric(df['FinishTime'], errors='coerce').fillna(0)
    zero_time = df[(df['FinishTime_Num'] <= 0) & (df['Place'] != 'DNF') & (df['Place'] != 'None')].shape[0]
    
    if zero_time > 0:
        issues.append(f"Found {zero_time} entries with FinishTime <= 0 (excluding DNFs).")
        
    if issues:
        print("\n[VALIDATION FAILED] Issues found:")
        for issue in issues:
            print(f" - {issue}")
            
        # Show sample of bad data
        print("\nSample of incomplete rows:")
        bad_rows = df[df['FinishTime_Num'] <= 0].head(5)
        print(bad_rows[['TrackName', 'RaceNumber', 'GreyhoundName', 'Place', 'FinishTime']])
        
        print("\n[ANALYSIS] Missing Data by Track:")
        # Group by Track and count nulls in FinishTime
        missing_df = df[df['FinishTime_Num'] <= 0]
        if not missing_df.empty:
            track_counts = missing_df['TrackName'].value_counts()
            print(track_counts)
    else:
        print("\n[SUCCESS] Data validation passed! All critical columns populated.")
        print(f"Checked {len(df)} entries.")
        print(df.head())

if __name__ == "__main__":
    validate_yesterday_data()
