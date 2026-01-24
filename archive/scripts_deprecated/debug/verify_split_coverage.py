import sqlite3
import pandas as pd

def check_coverage():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Checking Database Coverage:")
    
    # 1. Total Rows
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries")
    total = cursor.fetchone()[0]
    print(f"Total Entries: {total:,}")
    
    # 2. Valid Splits
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE Split IS NOT NULL AND Split != ''")
    splits = cursor.fetchone()[0]
    print(f"Entries with Split: {splits:,}")
    
    # 3. Valid Pace (FinishTimeBenchmarkLengths)
    cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE FinishTimeBenchmarkLengths IS NOT NULL")
    pace = cursor.fetchone()[0]
    print(f"Entries with Pace: {pace:,}")
    
    # 4. Difference
    diff = pace - splits
    print(f"Difference (Pace - Split): {diff:,}")
    
    pct = (splits / pace) * 100 if pace > 0 else 0
    print(f"Split Coverage relative to Pace: {pct:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    check_coverage()
