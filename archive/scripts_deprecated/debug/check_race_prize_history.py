import sqlite3
import pandas as pd

def check_race_prize():
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("Checking Race Prize Money Density (Historical)")
    print(f"{'Year':<6} {'Total Runs':<12} {'With RacePrize > 0':<20} {'%':<8}")
    print("-" * 60)
    
    for year in range(2020, 2026):
        query = f'''
        SELECT 
            COUNT(*) as Total,
            SUM(CASE WHEN ge.PrizeMoney > 0 THEN 1 ELSE 0 END) as HasPrize
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE strftime('%Y', rm.MeetingDate) = '{year}'
        '''
        
        row = conn.execute(query).fetchone()
        total = row[0]
        has_prize = row[1] if row[1] else 0
        pct = (has_prize / total * 100) if total > 0 else 0
        
        print(f"{year:<6} {total:<12,} {has_prize:<20,} {pct:<8.1f}%")
        
    conn.close()

if __name__ == "__main__":
    check_race_prize()
