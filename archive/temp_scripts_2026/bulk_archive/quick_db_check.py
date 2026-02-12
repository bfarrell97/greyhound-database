
import sqlite3
from datetime import datetime

def check_db():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("\n--- DATABASE DIAGNOSTIC ---")
    
    # 1. Latest data check
    latest_date = cursor.execute("SELECT max(MeetingDate) FROM RaceMeetings").fetchone()[0]
    print(f"Latest Date in DB: {latest_date}")
    
    # 2. Check for today (Dec 24)
    today = "2025-12-24"
    count_today = cursor.execute(f"SELECT count(*) FROM RaceMeetings WHERE MeetingDate = '{today}'").fetchone()[0]
    print(f"Meetings for {today}: {count_today}")
    
    if count_today > 0:
        # Check for races
        races = cursor.execute(f"SELECT count(*) FROM Races r JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate = '{today}'").fetchone()[0]
        print(f"Races for {today}: {races}")
        
        # Check for runners
        runners = cursor.execute(f"SELECT count(*) FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate = '{today}'").fetchone()[0]
        print(f"Runners for {today}: {runners}")
        
        # Sample one race
        sample = cursor.execute(f"SELECT r.RaceTime, t.TrackName FROM Races r JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID JOIN Tracks t ON rm.TrackID = t.TrackID WHERE rm.MeetingDate = '{today}' LIMIT 1").fetchone()
        if sample:
            print(f"Sample Race today: {sample[0]} at {sample[1]}")
    else:
        print("\n[!] CRITICAL: Today's schedule IS NOT in the database.")
        print("    This is why Alpha signals are not showing.")
        print("    You need to go to the 'Scrape Data' tab and run a fresh scrape.")

    conn.close()

if __name__ == "__main__":
    check_db()
