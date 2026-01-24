import sqlite3
import pandas as pd
from datetime import datetime

def analyze_track_bias():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Query today's data (Results + Live Prices)
    # We want to compare Price5Min vs BSP or Opening Price
    query = """
    SELECT 
        t.TrackName, 
        t.State,
        COUNT(*) as Runners,
        AVG(CAST(ge.Price5Min AS FLOAT)) as AvgPrice,
        AVG(CASE WHEN ge.BSP > 0 THEN ge.Price5Min / ge.BSP ELSE NULL END) as MoveRatio_vs_BSP,
        AVG(CASE WHEN ge.Price60Min > 0 THEN ge.Price5Min / ge.Price60Min ELSE NULL END) as MoveRatio_vs_Open,
        SUM(CASE WHEN ge.Price5Min < ge.Price60Min * 0.9 THEN 1 ELSE 0 END) as Steamers,
        SUM(CASE WHEN ge.Price5Min > ge.Price60Min * 1.1 THEN 1 ELSE 0 END) as Drifters
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = DATE('now')
      AND ge.Price5Min > 0
    GROUP BY t.TrackName
    ORDER BY Tracks.TrackName
    """
    
    try:
        # Note: SQLite date('now') depends on system time. Python datetime is safer if timezone issues.
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"Analyzing Track Bias for Today: {today}")
        
        # Simplified query to just get raw data and process in Pandas
        df = pd.read_sql_query("""
            SELECT 
                t.TrackName, 
                ge.Price5Min,
                ge.Price60Min,
                ge.Price30Min,
                ge.BSP
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE rm.MeetingDate >= ?
              AND ge.Price5Min > 0
        """, conn, params=(today,))
        
        if len(df) == 0:
            print("No data found for today.")
            return

        # Calculate Steam/Drift
        # Use Price30Min or Price60Min as baseline
        df['OpenPrice'] = df['Price60Min'].combine_first(df['Price30Min'])
        df = df.dropna(subset=['OpenPrice'])
        
        df['Ratio'] = df['Price5Min'] / df['OpenPrice']
        
        print(f"\n{'Track':<20} | {'Runners':<8} | {'Avg Ratio':<10} | {'% Steam':<10} | {'% Drift':<10}")
        print("-" * 70)
        
        for track, group in df.groupby('TrackName'):
            count = len(group)
            avg_ratio = group['Ratio'].mean()
            # Ratio < 1.0 = Steam (Price Dropped)
            # Ratio > 1.0 = Drift (Price Rose)
            pct_steam = (group['Ratio'] < 0.95).mean() * 100
            pct_drift = (group['Ratio'] > 1.05).mean() * 100
            
            print(f"{track:<20} | {count:<8} | {avg_ratio:>9.3f}x | {pct_steam:>9.1f}% | {pct_drift:>9.1f}%")
            
        print("-" * 70)
        print("Avg Ratio < 1.0 means Track tends to STEAM.")
        print("Avg Ratio > 1.0 means Track tends to DRIFT.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_track_bias()
