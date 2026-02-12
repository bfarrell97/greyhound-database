import pandas as pd
import sqlite3
import sys
import os

def check():
    print("Checking 'THOUGHT OF THAT' on 2025-12-23...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Get Dog ID
    dog_name = 'THOUGHT OF THAT'
    
    query = f"""
    SELECT 
        ge.Split, ge.FinishTime, ge.InRun, ge.Box, ge.Position,
        t.TrackName, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = '2025-12-23'
    AND ge.GreyhoundID IN (SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName = '{dog_name}')
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No DB record found for this dog/date.")
    else:
        print("DB Data:")
        print(df.to_string(index=False))
        print("\nTopaz Data (Reference):")
        print("Split: 6.71")
        print("Result: 26.502")
        print("PIR: 123")

if __name__ == "__main__":
    check()
