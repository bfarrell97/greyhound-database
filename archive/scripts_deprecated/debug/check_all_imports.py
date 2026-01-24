import sqlite3
import pandas as pd

def check_all_imports():
    conn = sqlite3.connect('greyhound_racing.db')
    
    date = '2025-12-13'
    print(f"--- IMPORT STATUS FOR {date} ---")
    
    query = f"""
    SELECT 
        t.TrackName,
        COUNT(DISTINCT r.RaceID) as TotalRaces,
        COUNT(ge.EntryID) as TotalEntries,
        SUM(CASE WHEN ge.StartingPrice IS NOT NULL THEN 1 ELSE 0 END) as EntriesWithOdds,
        SUM(CASE WHEN ge.Box IS NOT NULL THEN 1 ELSE 0 END) as EntriesWithBox
    FROM Races r 
    JOIN RaceMeetings rm ON r.MeetingID=rm.MeetingID 
    JOIN Tracks t ON rm.TrackID=t.TrackID
    LEFT JOIN GreyhoundEntries ge ON r.RaceID=ge.RaceID
    WHERE rm.MeetingDate='{date}'
    GROUP BY t.TrackName
    ORDER BY t.TrackName
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("No races found for today.")
        else:
            # Calculate percentages for better readability
            df['OddsCoverage'] = (df['EntriesWithOdds'] / df['TotalEntries'] * 100).round(1)
            df['BoxCoverage'] = (df['EntriesWithBox'] / df['TotalEntries'] * 100).round(1)
            
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 1000)
            output = df.to_string(index=False)
            print(output)
            
            with open('logs/import_status.txt', 'w') as f:
                f.write(output)
            print("Wrote detailed output to logs/import_status.txt")
            
            # Check for any complete failures
            failed = df[df['EntriesWithOdds'] == 0]
            if not failed.empty:
                print("\n[WARNING] The following tracks have ZERO odds data:")
                print(failed['TrackName'].tolist())
                
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    check_all_imports()
