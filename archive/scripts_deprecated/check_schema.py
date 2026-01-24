"""Quick schema check"""
import sqlite3
conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

print("=== GreyhoundEntries ===")
c.execute("PRAGMA table_info(GreyhoundEntries)")
for row in c.fetchall():
    print(f"  {row[1]}: {row[2]}")

print("\n=== Greyhounds ===")
c.execute("PRAGMA table_info(Greyhounds)")
for row in c.fetchall():
    print(f"  {row[1]}: {row[2]}")

print("\n=== Races ===")
c.execute("PRAGMA table_info(Races)")
for row in c.fetchall():
    print(f"  {row[1]}: {row[2]}")

print("\n=== RaceMeetings ===")
c.execute("PRAGMA table_info(RaceMeetings)")
for row in c.fetchall():
    print(f"  {row[1]}: {row[2]}")

print("\n=== Sample RunningPosition values ===")
c.execute("SELECT DISTINCT RunningPosition FROM GreyhoundEntries WHERE RunningPosition IS NOT NULL LIMIT 20")
for row in c.fetchall():
    print(f"  {row[0]}")

print("\n=== Sample IncomingGrade values ===")
c.execute("SELECT DISTINCT IncomingGrade FROM GreyhoundEntries WHERE IncomingGrade IS NOT NULL LIMIT 20")
for row in c.fetchall():
    print(f"  {row[0]}")

conn.close()
