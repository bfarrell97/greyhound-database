"""Debug percentage of entries needing BSP by track"""
import sqlite3
conn = sqlite3.connect('greyhound_racing.db')

print('November 2025 - Entries with/without BSP by track:')
query = """
SELECT t.TrackName, 
       SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP,
       SUM(CASE WHEN ge.BSP IS NULL THEN 1 ELSE 0 END) as NeedsBSP,
       COUNT(*) as Total
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate <= '2025-11-30'
GROUP BY t.TrackName
ORDER BY NeedsBSP DESC
LIMIT 20
"""
print(f"{'Track':<20} {'HasBSP':>8} {'NeedsBSP':>10} {'Total':>8} {'%Need':>8}")
print("-"*56)
for row in conn.execute(query).fetchall():
    track, has, needs, total = row
    pct = (needs / total * 100) if total > 0 else 0
    print(f"{track:<20} {has:>8} {needs:>10} {total:>8} {pct:>7.1f}%")

print("\n\nSample entries needing BSP (November, specific track):")
query = """
SELECT rm.MeetingDate, t.TrackName, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.BSP IS NULL 
  AND rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate <= '2025-11-30'
  AND t.TrackName = 'Sandown'
LIMIT 10
"""
for row in conn.execute(query).fetchall():
    print(f"  {row}")

conn.close()
