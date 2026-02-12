import sqlite3
import pandas as pd

def check_30min():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check Columns explicitly
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    cols = [r[1] for r in cursor.fetchall()]
    
    target = 'Price30Min'
    if target in cols:
        print(f"✅ Column '{target}' EXISTS in schema.")
        
        # Check Coverage for 2025
        query = f"""
        SELECT 
            COUNT(*) as TotalRows,
            COUNT({target}) as Count30Min,
            AVG({target}) as AvgPrice
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate >= '2025-01-01'
        AND ge.Box IS NOT NULL
        """
        try:
            df = pd.read_sql_query(query, conn)
            print("\nJan 2025 Coverage:")
            print(df)
        except Exception as e:
            print(f"Query failed: {e}")
            
    else:
        print(f"❌ Column '{target}' does NOT exist in schema.")
        print("Available 'Price' columns:")
        print([c for c in cols if 'Price' in c])
        
    conn.close()

if __name__ == "__main__":
    check_30min()
