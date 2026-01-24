import sqlite3
import pandas as pd

def check_track_splits():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("Checking Split Coverage by Track (Last 5 Years)")
    print("-" * 70)
    print(f"{'TrackName':<25} {'Runs':<10} {'WithSplit':<10} {'Split %':<10}")
    print("-" * 70)
    
    query = '''
    SELECT 
        t.TrackName,
        COUNT(*) as Runs,
        SUM(CASE WHEN ge.Split IS NOT NULL AND ge.Split != '' THEN 1 ELSE 0 END) as WithSplit
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2020-01-01'
    GROUP BY t.TrackName
    HAVING Runs > 1000
    ORDER BY (CAST(WithSplit AS FLOAT) / Runs) ASC, Runs DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    
    df['Pct'] = (df['WithSplit'] / df['Runs']) * 100
    
    # Show tracks with Low Coverage
    low = df[df['Pct'] < 10]
    print(f"Tracks with < 10% Split Data: {len(low)}")
    if len(low) > 0:
        print(low.head(20).to_string(index=False))
        
    print("-" * 70)
    
    # Show summary
    total_tracks = len(df)
    zero_split_tracks = len(df[df['Pct'] == 0])
    print(f"Total Tracks analyzed: {total_tracks}")
    print(f"Tracks with ZERO splits: {zero_split_tracks}")
    
    conn.close()

if __name__ == "__main__":
    check_track_splits()
