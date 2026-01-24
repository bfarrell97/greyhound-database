"""Check tracks with lowest BSP coverage"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')

print("Tracks with LOWEST BSP coverage:")
print("="*70)

query = """
SELECT t.TrackName,
       SUM(CASE WHEN ge.BSP IS NOT NULL THEN 1 ELSE 0 END) as HasBSP,
       SUM(CASE WHEN ge.BSP IS NULL THEN 1 ELSE 0 END) as NeedsBSP,
       COUNT(*) as Total
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
GROUP BY t.TrackName
ORDER BY (HasBSP * 1.0 / Total) ASC
"""

print(f"{'Track':<25} {'HasBSP':>10} {'NeedsBSP':>10} {'Total':>10} {'Coverage':>10}")
print("-"*70)

for row in c.execute(query).fetchall():
    name, has, needs, total = row
    pct = has/total*100 if total > 0 else 0
    print(f"{name:<25} {has:>10,} {needs:>10,} {total:>10,} {pct:>9.1f}%")

print("\n" + "="*70)
total_has = c.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL").fetchone()[0]
total_all = c.execute("SELECT COUNT(*) FROM GreyhoundEntries").fetchone()[0]
print(f"OVERALL: {total_has:,} / {total_all:,} = {total_has/total_all*100:.1f}% coverage")
