import pandas as pd
import sqlite3
import sys

def check_historical_coverage():
    print("Checking Historical Price Coverage (Yearly Breakdown)...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    query = """
    SELECT 
        strftime('%Y', rm.MeetingDate) as Year,
        COUNT(*) as TotalRaces,
        COUNT(ge.Price5Min) as Count_5Min,
        COUNT(ge.Price10Min) as Count_10Min,
        COUNT(ge.PriceOpen) as Count_Open
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    GROUP BY strftime('%Y', rm.MeetingDate)
    ORDER BY Year
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n" + "="*60)
    print(f"{'Year':<6} | {'Total':<8} | {'5Min%':<8} | {'10Min%':<8} | {'Open%':<8}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        yr = row['Year']
        tot = row['TotalRaces']
        p5 = row['Count_5Min']
        p10 = row['Count_10Min']
        popen = row['Count_Open']
        
        print(f"{yr:<6} | {tot:<8} | {p5/tot*100:>6.1f}% | {p10/tot*100:>6.1f}% | {popen/tot*100:>6.1f}%")

if __name__ == "__main__":
    check_historical_coverage()
