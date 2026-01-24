import sqlite3
import datetime

db_path = 'greyhound_racing.db'
today = '2025-12-20'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Reset prices for ALL December 2025 races
    print(f"Clearing invalid prices for December 2025...")
    
    query = """
    UPDATE GreyhoundEntries 
    SET Price60Min=NULL, Price30Min=NULL, Price15Min=NULL, Price10Min=NULL, 
        Price5Min=NULL, Price2Min=NULL, Price1Min=NULL
    WHERE RaceID IN (
        SELECT r.RaceID 
        FROM Races r
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate LIKE '2025-12-%'
    )
    """
    
    cursor.execute(query)
    print(f"Cleared prices for {cursor.rowcount} entries.")
    
    conn.commit()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
