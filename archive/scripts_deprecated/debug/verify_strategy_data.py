import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("Verifying data compliance for Pace Strategy...")
    print(f"{'Year':<6} {'Total Runs':<12} {'With Split':<12} {'With Prize':<12} {'Fully Valid':<12} {'Valid %':<8}")
    print("-" * 70)
    
    for year in [2023, 2024, 2025]:
        query = f'''
        SELECT 
            COUNT(*) as Total,
            SUM(CASE WHEN ge.Split IS NOT NULL AND ge.Split != '' THEN 1 ELSE 0 END) as WithSplit,
            SUM(CASE WHEN ge.CareerPrizeMoney IS NOT NULL THEN 1 ELSE 0 END) as WithPrize,
            SUM(CASE WHEN 
                ge.Position IS NOT NULL AND 
                ge.StartingPrice IS NOT NULL AND 
                ge.Split IS NOT NULL AND ge.Split != '' AND
                ge.CareerPrizeMoney IS NOT NULL
                THEN 1 ELSE 0 END) as FullyValid
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE strftime('%Y', rm.MeetingDate) = '{year}'
        '''
        
        row = conn.execute(query).fetchone()
        total = row[0]
        valid = row[3]
        pct = (valid / total * 100) if total > 0 else 0
        
        print(f"{year:<6} {total:<12,} {row[1]:<12,} {row[2]:<12,} {valid:<12,} {pct:<8.1f}%")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")
