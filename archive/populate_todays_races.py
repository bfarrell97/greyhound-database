"""
Populate UpcomingBettingRaces with today's races from completed races data
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def populate_todays_races():
    """Fetch today's completed races and add to UpcomingBettingRaces"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get the most recent date with race data
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(MeetingDate) FROM RaceMeetings')
    latest_date = cursor.fetchone()[0]
    
    if not latest_date:
        print("No race data found in database")
        conn.close()
        return
    
    print(f"Using most recent race date: {latest_date}")
    query = """
    SELECT DISTINCT
        rm.MeetingDate,
        t.TrackName,
        r.RaceNumber,
        r.Distance,
        r.RaceID
    FROM Races r
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = ?
    ORDER BY r.RaceNumber
    """
    
    races = pd.read_sql_query(query, conn, params=(latest_date,))
    
    if len(races) == 0:
        print(f"No races found for {today}")
        conn.close()
        return
    
    print(f"Found {len(races)} races for {today}")
    
    # Insert into UpcomingBettingRaces
    cursor = conn.cursor()
    
    for _, race in races.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO UpcomingBettingRaces 
                (MeetingDate, TrackName, RaceNumber, Distance, LastUpdated)
                VALUES (?, ?, ?, ?, ?)
            """, (
                race['MeetingDate'],
                race['TrackName'],
                int(race['RaceNumber']),
                int(race['Distance']),
                datetime.now().isoformat()
            ))
        except Exception as e:
            print(f"Error inserting race: {e}")
    
    conn.commit()
    
    # Now add runners with odds from completed races
    runner_query = """
    SELECT
        ge.RaceID,
        g.GreyhoundName,
        ge.Box as BoxNumber,
        ge.Weight,
        ge.StartingPrice as CurrentOdds
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = ?
      AND ge.StartingPrice IS NOT NULL
    """
    
    runners = pd.read_sql_query(runner_query, conn, params=(today,))
    
    if len(runners) == 0:
        print(f"No runners with odds found for {today}")
        conn.close()
        return
    
    print(f"Adding {len(runners)} runners with odds")
    
    for _, runner in runners.iterrows():
        try:
            # Get the UpcomingBettingRaceID for this race
            cursor.execute("""
                SELECT UpcomingBettingRaceID FROM UpcomingBettingRaces 
                WHERE MeetingDate = ? AND RaceNumber = ?
            """, (today, int(runner.get('RaceNumber', 1))))
            
            result = cursor.fetchone()
            if result:
                race_id = result[0]
                
                cursor.execute("""
                    INSERT OR IGNORE INTO UpcomingBettingRunners
                    (UpcomingBettingRaceID, GreyhoundName, BoxNumber, Weight, CurrentOdds, LastUpdated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    runner['GreyhoundName'],
                    int(runner['BoxNumber']) if pd.notna(runner['BoxNumber']) else None,
                    float(runner['Weight']) if pd.notna(runner['Weight']) else None,
                    float(runner['CurrentOdds']) if pd.notna(runner['CurrentOdds']) else None,
                    datetime.now().isoformat()
                ))
        except Exception as e:
            print(f"Error adding runner: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✓ Populated UpcomingBettingRaces with {len(races)} races for {today}")

if __name__ == "__main__":
    populate_todays_races()
