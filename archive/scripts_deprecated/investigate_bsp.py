"""Deep investigation into why 50% of entries don't have BSP"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')

print("="*70)
print("DEEP DIVE: Why 50% of entries are missing BSP")
print("="*70)

# 1. Check if it's a year problem
print("\n1. BSP COVERAGE BY YEAR:")
q = """
SELECT strftime('%Y', rm.MeetingDate) as Year,
       COUNT(*) as Total,
       SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
GROUP BY Year ORDER BY Year
"""
print(f"{'Year':<6} {'Total':>12} {'HasBSP':>10} {'Coverage':>10}")
print("-"*40)
for row in conn.execute(q):
    pct = row[2]/row[1]*100 if row[1] > 0 else 0
    print(f"{row[0]:<6} {row[1]:>12,} {row[2]:>10,} {pct:>9.1f}%")

# 2. Check venue names in BSP files vs DB
print("\n2. SAMPLE: Entries without BSP from 2022 (highest coverage year)")
q = """
SELECT UPPER(t.TrackName), rm.MeetingDate, g.GreyhoundName
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.BSP IS NULL 
  AND rm.MeetingDate BETWEEN '2022-01-01' AND '2022-12-31'
LIMIT 20
"""
print(f"{'Track':<25} {'Date':<12} {'Dog':<30}")
print("-"*70)
for row in conn.execute(q):
    print(f"{row[0]:<25} {row[1]:<12} {row[2]:<30}")

# 3. Check if the issue is specific tracks
print("\n3. TRACKS WITH MOST MISSING BSP IN 2022:")
q = """
SELECT UPPER(t.TrackName) as Track,
       COUNT(*) as NeedsBSP
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.BSP IS NULL 
  AND rm.MeetingDate BETWEEN '2022-01-01' AND '2022-12-31'
GROUP BY Track ORDER BY NeedsBSP DESC LIMIT 10
"""
print(f"{'Track':<25} {'Needs BSP':>12}")
print("-"*40)
for row in conn.execute(q):
    print(f"{row[0]:<25} {row[1]:>12,}")

# 4. Check if entries with BSP have matching dates with entries without
print("\n4. SAMPLE DATES WITH MIXED BSP COVERAGE (same track, same day):")
q = """
SELECT UPPER(t.TrackName), rm.MeetingDate,
       SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP,
       SUM(CASE WHEN ge.BSP IS NULL THEN 1 ELSE 0 END) as NeedsBSP
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate BETWEEN '2022-06-01' AND '2022-06-30'
GROUP BY t.TrackName, rm.MeetingDate
HAVING HasBSP > 0 AND NeedsBSP > 0
ORDER BY NeedsBSP DESC LIMIT 10
"""
print(f"{'Track':<25} {'Date':<12} {'Has BSP':>10} {'Needs BSP':>12}")
print("-"*65)
for row in conn.execute(q):
    print(f"{row[0]:<25} {row[1]:<12} {row[2]:>10} {row[3]:>12}")

conn.close()
