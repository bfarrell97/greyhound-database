import sqlite3
import pandas as pd

def check():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Check Schema
    cursor.execute('PRAGMA table_info(GreyhoundEntries)')
    cols = [info[1] for info in cursor.fetchall()]
    print("Columns in GreyhoundEntries:")
    print(cols)
    
    # Check if 'PIR' or 'Sectional' exists
    pir_col = None
    for c in cols:
        if 'pir' in c.lower() or 'sect' in c.lower() or 'runhome' in c.lower():
            print(f"Found potential PIR/Sectional column: {c}")
            if 'Topaz' not in c:
                pir_col = c
                
    # Check Coverage for Split and Found PIR col
    select_pir = f", COUNT(ge.{pir_col}) as PIRCount" if pir_col else ""
    
    query = f"""
    SELECT 
        strftime('%Y', rm.MeetingDate) as Year, 
        COUNT(*) as TotalRuns, 
        COUNT(ge.Split) as SplitCount
        {select_pir}
    FROM GreyhoundEntries ge 
    JOIN Races r ON ge.RaceID = r.RaceID 
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
    GROUP BY Year
    ORDER BY Year
    """
    
    print(f"\nQuerying Coverage...")
    try:
        df = pd.read_sql_query(query, conn)
        print(df)
        
        # Calculate percentages
        df['SplitPct'] = df['SplitCount'] / df['TotalRuns'] * 100
        if pir_col:
            df['PIRPct'] = df['PIRCount'] / df['TotalRuns'] * 100
        print("\nPercentages:")
        print(df)
        
    except Exception as e:
        print(f"Query Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    check()
