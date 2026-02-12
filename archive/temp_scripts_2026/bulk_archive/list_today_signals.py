import sqlite3
import pandas as pd
from datetime import datetime

# DB Path
DB_PATH = 'greyhound_racing.db'
TODAY = '2025-12-31'

def list_signals():
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check LiveBets Table
        query = f"""
        SELECT 
            PlacedDate, RaceTime, SelectionName as Dog, 
            Side, Price, Status, Result
        FROM LiveBets 
        WHERE (PlacedDate LIKE '{TODAY}%' OR MeetingDate = '{TODAY}')
        AND Side = 'BACK'
        ORDER BY PlacedDate DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print(f"No BACK bets found for {TODAY}.")
            return

        print(f"\n=== BACK SIGNALS FOR {TODAY} ===")
        # formatting
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_signals()
