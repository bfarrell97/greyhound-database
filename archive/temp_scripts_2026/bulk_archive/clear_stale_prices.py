"""
Clear stale prices for future races from GreyhoundEntries.
This removes contaminated Place market prices captured before the scraper fix.
"""
import sqlite3
from datetime import datetime

def clear_future_prices():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    # Get current time
    now = datetime.now().strftime('%H:%M')
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f'Current time: {now}')
    print(f'Today: {today}')
    
    # Find price columns in GreyhoundEntries
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    columns = [col[1] for col in cursor.fetchall()]
    price_cols = [c for c in columns if 'Price' in c or 'Lay' in c]
    print(f'Price columns found: {price_cols}')
    
    # Count entries for today's future races via Races/RaceMeetings join
    cursor.execute("""
        SELECT COUNT(*) FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate = ? AND r.RaceTime > ?
    """, (today, now))
    count = cursor.fetchone()[0]
    print(f'Entries to clear (future races today): {count}')
    
    # Set price columns to NULL for future races
    for col in price_cols:
        cursor.execute(f"""
            UPDATE GreyhoundEntries 
            SET {col} = NULL 
            WHERE RaceID IN (
                SELECT r.RaceID FROM Races r
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                WHERE rm.MeetingDate = ? AND r.RaceTime > ?
            )
        """, (today, now))
        
    conn.commit()
    print(f'\n✓ Cleared {len(price_cols)} price columns for {count} future race entries!')
    print('Restart run.py - the scraper will recapture correct WIN-only prices.')
    conn.close()

if __name__ == '__main__':
    clear_future_prices()
