"""Debug track matching issue"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')

# Get upcoming tracks
upcoming_query = """
    SELECT DISTINCT TrackName, TrackCode
    FROM UpcomingBettingRaces
    ORDER BY TrackName
"""
upcoming_df = pd.read_sql_query(upcoming_query, conn)

# Get historical tracks
historical_query = "SELECT TrackID, TrackName, TrackKey FROM Tracks"
historical_df = pd.read_sql_query(historical_query, conn)

print("\nUPCOMING TRACKS:")
print("-" * 60)
for _, row in upcoming_df.iterrows():
    print(f"  {row['TrackName']:<30} (code: {row['TrackCode']})")

print("\n\nHISTORICAL TRACKS (sample):")
print("-" * 60)
for _, row in historical_df.head(20).iterrows():
    print(f"  ID:{row['TrackID']:<4} {row['TrackName']:<30} (key: {row['TrackKey']})")

# Check matching logic
track_mapping = {}
for _, track_row in historical_df.iterrows():
    track_mapping[track_row['TrackName']] = track_row['TrackID']
    if '_' in str(track_row['TrackKey']):
        track_code = track_row['TrackKey'].split('_')[0]
        track_mapping[track_code] = track_row['TrackID']

print("\n\nMATCHING RESULTS:")
print("-" * 60)
for _, row in upcoming_df.iterrows():
    track_name = row['TrackName']
    track_code = row['TrackCode']

    name_match = track_mapping.get(track_name)
    code_match = track_mapping.get(track_code)

    if name_match:
        print(f"  {track_name:<30} MATCHED by name -> TrackID {name_match}")
    elif code_match:
        print(f"  {track_name:<30} MATCHED by code -> TrackID {code_match}")
    else:
        print(f"  {track_name:<30} NOT MATCHED (code: {track_code})")

conn.close()
