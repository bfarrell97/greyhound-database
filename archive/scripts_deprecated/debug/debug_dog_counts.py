import sqlite3
import pandas as pd

def debug_dog_counts():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("DEBUG: Unique Dog Counts (2020-2025)")
    print("-" * 50)
    
    # 1. Total Unique Dogs
    cursor.execute("""
        SELECT COUNT(DISTINCT ge.GreyhoundID) 
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate >= '2020-01-01' AND rm.MeetingDate <= '2025-12-09'
    """)
    total_dogs = cursor.fetchone()[0]
    print(f"Total Unique Dogs raced: {total_dogs:,}")
    
    # 2. Dogs with >= 1 Valid Split
    cursor.execute("""
        SELECT COUNT(DISTINCT ge.GreyhoundID) 
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate >= '2020-01-01' AND rm.MeetingDate <= '2025-12-09'
          AND ge.Split IS NOT NULL AND ge.Split != ''
    """)
    dogs_with_any_split = cursor.fetchone()[0]
    print(f"Dogs with >= 1 Valid Split: {dogs_with_any_split:,}")
    
    # 3. Dogs with >= 6 Valid Splits (Requirement for HistAvgSplit)
    # Why 6? Because shift(1) + rolling(5) min_periods(5) needs 5 PRIOR races.
    # So the 6th race is the first one with a value.
    query_6 = """
        SELECT COUNT(*) FROM (
            SELECT ge.GreyhoundID
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE rm.MeetingDate >= '2020-01-01' AND rm.MeetingDate <= '2025-12-09'
              AND ge.Split IS NOT NULL AND ge.Split != ''
            GROUP BY ge.GreyhoundID
            HAVING COUNT(*) >= 6
        )
    """
    cursor.execute(query_6)
    dogs_qualifying = cursor.fetchone()[0]
    print(f"Dogs with >= 6 Valid Splits (Qualify for history): {dogs_qualifying:,}")
    
    print("-" * 50)
    print(f"Compare 'Dogs Qualify' ({dogs_qualifying:,}) with script output (~44,877).")
    
    conn.close()

if __name__ == "__main__":
    debug_dog_counts()
