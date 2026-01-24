"""Check upcoming races distribution"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

# Upcoming races by track
cursor.execute("""
    SELECT TrackName, COUNT(*) as races
    FROM UpcomingBettingRaces
    GROUP BY TrackName
    ORDER BY races DESC
""")

print("\nUpcoming races by track:")
print("-" * 50)
for row in cursor.fetchall():
    print(f"{row[0]:<30} {row[1]:>3} races")

# Check prediction output - see what tracks have predictions
cursor.execute("""
    SELECT DISTINCT ubr_track.TrackName, COUNT(*) as runner_count
    FROM UpcomingBettingRunners ubr
    JOIN UpcomingBettingRaces ubr_track ON ubr.UpcomingBettingRaceID = ubr_track.UpcomingBettingRaceID
    JOIN Greyhounds g ON UPPER(TRIM(ubr.GreyhoundName)) = UPPER(TRIM(g.GreyhoundName))
    GROUP BY ubr_track.TrackName
    ORDER BY runner_count DESC
""")

print("\nUpcoming runners WITH historical greyhound match by track:")
print("-" * 50)
for row in cursor.fetchall():
    print(f"{row[0]:<30} {row[1]:>3} runners")

conn.close()
