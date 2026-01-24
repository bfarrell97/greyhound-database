"""Check overall BSP coverage by year"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')

print("BSP Coverage by Year:")
print("="*50)

query = """
SELECT strftime('%Y', rm.MeetingDate) as Year,
       COUNT(*) as Total,
       SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP,
       SUM(CASE WHEN ge.BSP IS NULL THEN 1 ELSE 0 END) as NeedsBSP
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2020-01-01'
GROUP BY Year
ORDER BY Year
"""

print(f"{'Year':<6} {'Total':>10} {'HasBSP':>10} {'NeedsBSP':>10} {'Coverage':>10}")
print("-"*50)
for row in c.execute(query).fetchall():
    year, total, has, needs = row
    pct = (has / total * 100) if total > 0 else 0
    print(f"{year:<6} {total:>10,} {has:>10,} {needs:>10,} {pct:>9.1f}%")

# Overall
overall = c.execute("""
SELECT COUNT(*) as Total,
       SUM(CASE WHEN BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP
FROM GreyhoundEntries
""").fetchone()
print("-"*50)
print(f"{'TOTAL':<6} {overall[0]:>10,} {overall[1]:>10,} {overall[0]-overall[1]:>10,} {overall[1]/overall[0]*100:>9.1f}%")
