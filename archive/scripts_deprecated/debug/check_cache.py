
import sqlite3
import os

db_path = 'greyhound_racing.db'
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# dogs_to_check = [301543]
dogs_to_check = [326334, 1813698]

for dog_id in dogs_to_check:
    print(f"\nChecking cache for Dog {dog_id}...")
    cursor.execute("SELECT RaceID, length(FeaturesJSON), FeaturesJSON FROM DogFeatureCache WHERE DogID=?", (dog_id,))
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No cache entries found for Dog {dog_id}.")
    else:
        for r in rows:
            content_preview = r[2] if r[2] == 'null' else f"JSON (len={r[1]})"
            print(f"Found Entry: RaceID={r[0]}, Content={content_preview}")


# Also check Race info to confirm date
if rows:
    race_id = rows[0][0]
    cursor.execute("SELECT MeetingID, RaceNumber, RaceTime FROM Races WHERE RaceID=?", (race_id,))
    race = cursor.fetchone()
    if race:
        cursor.execute("SELECT MeetingDate, TrackID FROM RaceMeetings WHERE MeetingID=?", (race[0],))
        meeting = cursor.fetchone()
        print(f"Race Info: Date={meeting[0]}, RaceNum={race[1]}")
    else:
        print("Race ID not found in Races table??")

conn.close()
