"""Check BSP coverage by year"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')
print('BSP Coverage by Year:')
print('Year  | Total Entries | With BSP | Coverage')
print('-'*50)
for year in range(2020, 2026):
    q = f"""
    SELECT COUNT(*), SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END)
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE strftime('%Y', rm.MeetingDate) = '{year}'
    """
    total, with_bsp = c.execute(q).fetchone()
    pct = with_bsp/total*100 if total > 0 else 0
    print(f'{year}  | {total:>13,} | {with_bsp:>8,} | {pct:>6.1f}%')
