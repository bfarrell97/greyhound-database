"""Check what historical data we have"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

# Overall range (using RaceMeetings table)
c.execute('SELECT MIN(MeetingDate) as earliest, MAX(MeetingDate) as latest, COUNT(DISTINCT MeetingDate) as total_days FROM RaceMeetings')
earliest, latest, total_days = c.fetchone()
print(f"Historical data range: {earliest} to {latest}")
print(f"Total meeting days: {total_days:,}")

# Count races
c.execute('SELECT COUNT(*) FROM Races')
total_races = c.fetchone()[0]
print(f"Total races: {total_races:,}")

# Count entries
c.execute('SELECT COUNT(*) FROM GreyhoundEntries')
total_entries = c.fetchone()[0]
print(f"Total greyhound entries: {total_entries:,}")

# November 2025 data
c.execute('SELECT COUNT(DISTINCT rm.MeetingDate) FROM RaceMeetings rm WHERE rm.MeetingDate >= "2025-11-01" AND rm.MeetingDate < "2025-12-01"')
nov_days = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM Races r JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate >= "2025-11-01" AND rm.MeetingDate < "2025-12-01"')
nov_races = c.fetchone()[0]
print(f"\nNovember 2025: {nov_days} days, {nov_races} races")

if nov_days > 0:
    c.execute('''
        SELECT rm.MeetingDate, COUNT(DISTINCT r.RaceID) as races
        FROM RaceMeetings rm
        LEFT JOIN Races r ON r.MeetingID = rm.MeetingID
        WHERE rm.MeetingDate >= "2025-11-01" AND rm.MeetingDate < "2025-12-01"
        GROUP BY rm.MeetingDate
        ORDER BY rm.MeetingDate DESC
        LIMIT 10
    ''')
    print("\nRecent November 2025 dates:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]} races")
else:
    print("\n⚠️  NO NOVEMBER 2025 DATA FOUND!")
    print("We need to scrape November results to validate the model.")

# December 2025 data
c.execute('SELECT COUNT(DISTINCT rm.MeetingDate) FROM RaceMeetings rm WHERE rm.MeetingDate >= "2025-12-01" AND rm.MeetingDate < "2025-12-09"')
dec_days = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM Races r JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate >= "2025-12-01" AND rm.MeetingDate < "2025-12-09"')
dec_races = c.fetchone()[0]
print(f"\nDecember 2025 (up to Dec 8): {dec_days} days, {dec_races} races")

conn.close()
