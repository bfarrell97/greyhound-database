import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check AWESOME BINDI entries
    dog_id = 318894
    print(f"Checking entries for Dog ID {dog_id} (AWESOME BINDI)")
    
    query = f"SELECT * FROM GreyhoundEntries WHERE GreyhoundID = {dog_id}"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No entries found!")
    else:
        print(f"Found {len(df)} entries")
        print("First 5 entries (FirstSplitPosition):")
        print(df[['FirstSplitPosition', 'Position', 'RaceID']].head(5).to_string())
        
        # Check if they have splits
        valid_splits = df[df['FirstSplitPosition'].notna() & (df['FirstSplitPosition'] != '')]
        print(f"Entries with valid splits: {len(valid_splits)}")

    # Check Latest Meeting Date properly
    cursor.execute("SELECT MAX(MeetingDate) FROM RaceMeetings")
    print(f"Latest Meeting Date in DB: {cursor.fetchone()[0]}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
