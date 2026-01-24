"""Check which upcoming greyhounds don't have historical data"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

# Check match rate
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN g.GreyhoundID IS NOT NULL THEN 1 ELSE 0 END) as matched
    FROM UpcomingBettingRunners ubr
    LEFT JOIN Greyhounds g ON UPPER(TRIM(ubr.GreyhoundName)) = UPPER(TRIM(g.GreyhoundName))
""")
total, matched = cursor.fetchone()
print(f"\nUpcoming greyhound match rate:")
print(f"  Total upcoming runners: {total}")
print(f"  Matched to historical: {matched}")
print(f"  Not matched: {total - matched}")
print(f"  Match rate: {matched/total*100:.1f}%\n")

# Sample of unmatched greyhounds
cursor.execute("""
    SELECT DISTINCT ubr.GreyhoundName, ubrt.TrackName
    FROM UpcomingBettingRunners ubr
    JOIN UpcomingBettingRaces ubrt ON ubr.UpcomingBettingRaceID = ubrt.UpcomingBettingRaceID
    LEFT JOIN Greyhounds g ON UPPER(TRIM(ubr.GreyhoundName)) = UPPER(TRIM(g.GreyhoundName))
    WHERE g.GreyhoundID IS NULL
    LIMIT 20
""")

print("Sample of unmatched greyhounds:")
print("-" * 60)
for row in cursor.fetchall():
    print(f"{row[0]:<40} ({row[1]})")

conn.close()
