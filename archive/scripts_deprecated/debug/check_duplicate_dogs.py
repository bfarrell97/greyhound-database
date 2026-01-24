import sqlite3
import pandas as pd

def check_duplicates():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Check for Steely Shiraz
    print("--- Searching for 'Steely Shiraz' ---")
    df = pd.read_sql_query("SELECT * FROM Greyhounds WHERE GreyhoundName LIKE 'Steely Shiraz'", conn)
    print(df)
    
    print("\n--- Searching for 'STEELY SHIRAZ' ---")
    df2 = pd.read_sql_query("SELECT * FROM Greyhounds WHERE GreyhoundName LIKE 'STEELY SHIRAZ'", conn)
    print(df2)
    
    # Check Race 10 entries for these IDs
    print("\n--- Entries for Race 10 (Bendigo) ---")
    query = """
    SELECT ge.EntryID, ge.GreyhoundID, g.GreyhoundName, ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
    JOIN Races r ON ge.RaceID=r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID
    JOIN Tracks t ON rm.TrackID=t.TrackID
    WHERE rm.MeetingDate='2025-12-13' AND t.TrackName='Bendigo' AND r.RaceNumber=10
    """
    df3 = pd.read_sql_query(query, conn)
    print(df3)
    
    conn.close()

if __name__ == "__main__":
    check_duplicates()
