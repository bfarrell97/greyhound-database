import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

def analyze_november_data():
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT rm.MeetingDate, ge.Price5Min, ge.BSP, ge.GreyhoundID
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= '2025-07-01' 
    AND ge.Price5Min > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MoveRatio'] = df['Price5Min'] / df['BSP']
    df['Is_Drifter'] = (df['MoveRatio'] < 0.95).astype(int)
    df['Month'] = pd.to_datetime(df['MeetingDate']).dt.to_period('M')
    
    print("--- 📅 MONTHLY STATS (Drifter Target: Ratio < 0.95) ---")
    print(f"{'Month':<10} | {'Rows':<8} | {'Drift%':<8} | {'AvgRatio':<8} | {'AvgBSP':<8} | {'AvgPrice5':<8}")
    print("-" * 75)
    
    stats = df.groupby('Month').agg({
        'Is_Drifter': 'mean',
        'MoveRatio': 'mean',
        'BSP': 'mean',
        'Price5Min': 'mean',
        'GreyhoundID': 'count'
    })
    
    print(stats)

if __name__ == "__main__":
    analyze_november_data()
