"""
Debug script to check wins data in database
"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("CHECKING GREYHOUND ENTRIES")
print("=" * 80)

# Check a few entries
cursor.execute("""
    SELECT
        g.GreyhoundName,
        ge.RaceID,
        ge.Box,
        ge.Position,
        ge.FinishTime,
        ge.Margin,
        r.RaceNumber,
        r.Distance
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    ORDER BY r.RaceNumber
    LIMIT 20
""")

results = cursor.fetchall()
print(f"\nFound {len(results)} entries\n")

for row in results:
    name, race_id, box, position, finish_time, margin, race_num, distance = row
    print(f"Race {race_num} ({distance}m): {name} - Box {box}, Pos {position}, Time {finish_time}")

print("\n" + "=" * 80)
print("CHECKING GREYHOUND STATS")
print("=" * 80)

cursor.execute("""
    SELECT
        GreyhoundName,
        Starts,
        Wins,
        Seconds,
        Thirds,
        Prizemoney
    FROM Greyhounds
    ORDER BY GreyhoundName
    LIMIT 20
""")

results = cursor.fetchall()
print(f"\nFound {len(results)} greyhounds\n")

for row in results:
    name, starts, wins, seconds, thirds, prizemoney = row
    print(f"{name:30s} - Starts: {starts}, Wins: {wins}, 2nds: {seconds}, 3rds: {thirds}, Prize: ${prizemoney}")

print("\n" + "=" * 80)
print("CHECKING FOR POSITION 1 (WINS)")
print("=" * 80)

cursor.execute("""
    SELECT
        g.GreyhoundName,
        COUNT(*) as win_count
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE ge.Position = 1
    GROUP BY g.GreyhoundID, g.GreyhoundName
    ORDER BY win_count DESC
""")

results = cursor.fetchall()
print(f"\nGreyhounds with wins in entries:\n")

for row in results:
    name, win_count = row
    print(f"{name:30s} - {win_count} wins in entries")

conn.close()
