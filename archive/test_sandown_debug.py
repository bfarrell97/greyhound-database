"""
Debug why Sandown (SAP) races aren't getting predictions
"""
import sqlite3
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel

print("="*80)
print("SANDOWN (SAP) PREDICTION DEBUG")
print("="*80)

# Load model
m = GreyhoundMLModel()
m.load_model()

# Check what's in the upcoming betting table for Sandown
conn = sqlite3.connect('greyhound_racing.db')

print("\n1. Checking UpcomingBettingRaces for Sandown:")
c = conn.cursor()
c.execute("""
    SELECT UpcomingBettingRaceID, MeetingDate, TrackName, TrackCode, RaceNumber, Distance
    FROM UpcomingBettingRaces
    WHERE TrackName LIKE '%Sandown%'
    ORDER BY RaceNumber
""")
sandown_races = c.fetchall()
print(f"   Found {len(sandown_races)} Sandown races:")
for race in sandown_races[:3]:
    print(f"     Race {race[4]}: Date={race[1]}, Track=\"{race[2]}\", Code={race[3]}, Dist={race[5]}")

# Check what the model's query would return
print("\n2. Running model's query for upcoming races:")
query = """
SELECT
    ur.GreyhoundName,
    ubr.TrackName as CurrentTrack,
    ubr.TrackCode,
    ubr.RaceNumber,
    ur.BoxNumber as CurrentBox,
    ubr.Distance as CurrentDistance,
    AVG(ur.CurrentOdds) as CurrentOdds,
    ubr.MeetingDate,
    ubr.RaceTime
FROM UpcomingBettingRunners ur
JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID
WHERE ubr.MeetingDate = ?
GROUP BY ur.GreyhoundName, ubr.TrackName, ubr.RaceNumber, ur.BoxNumber, ubr.MeetingDate, ubr.TrackCode, ubr.RaceTime, ubr.Distance
ORDER BY ubr.RaceNumber, ur.BoxNumber
"""

df = pd.read_sql_query(query, conn, params=('2025-12-08',))
print(f"   Total entries: {len(df)}")
print(f"   Unique tracks: {df['CurrentTrack'].unique().tolist()}")

# Check after NZ/TAS filtering
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df['CurrentTrack'].isin(excluded_tracks)]
df = df[~df['CurrentTrack'].str.contains('NZ', na=False, case=False)]
print(f"   After NZ/TAS filter: {len(df)}")
print(f"   Unique tracks: {df['CurrentTrack'].unique().tolist()}")

# Check Sandown specifically
sandown_df = df[df['CurrentTrack'].str.contains('Sandown', na=False, case=False)]
print(f"\n3. Sandown entries after filter:")
print(f"   Count: {len(sandown_df)}")
if len(sandown_df) > 0:
    print(f"   Sample:")
    print(sandown_df[['GreyhoundName', 'CurrentTrack', 'RaceNumber', 'CurrentBox']].head())

# Check track matching
print("\n4. Track ID matching:")
track_query = "SELECT TrackID, TrackName, TrackKey FROM Tracks"
tracks_df = pd.read_sql_query(track_query, conn)

# Create mapping (same as model does)
track_mapping = {}
for _, track_row in tracks_df.iterrows():
    track_mapping[track_row['TrackName']] = track_row['TrackID']
    if '_' in str(track_row['TrackKey']):
        track_code = track_row['TrackKey'].split('_')[0]
        track_mapping[track_code] = track_row['TrackID']

# Try to match Sandown
sandown_track_name = "Sandown (SAP)"
if sandown_track_name in track_mapping:
    print(f"   [OK] '{sandown_track_name}' found in mapping -> TrackID={track_mapping[sandown_track_name]}")
else:
    print(f"   [ERROR] '{sandown_track_name}' NOT found in mapping")
    print(f"   Available tracks with 'Sandown':")
    for name, tid in track_mapping.items():
        if 'sandown' in name.lower():
            print(f"      '{name}' -> TrackID={tid}")

# Add TrackID to dataframe (same as model)
df['TrackID'] = df['CurrentTrack'].map(track_mapping)
df['TrackID'] = df['TrackID'].fillna(df['TrackCode'].map(track_mapping))

print(f"\n5. Track ID matching results:")
print(f"   Total entries: {len(df)}")
print(f"   Matched to TrackID: {df['TrackID'].notna().sum()}")
print(f"   Not matched: {df['TrackID'].isna().sum()}")

# Show which tracks didn't match
unmatched = df[df['TrackID'].isna()]
if len(unmatched) > 0:
    print(f"\n   Unmatched tracks:")
    for track in unmatched['CurrentTrack'].unique():
        count = len(unmatched[unmatched['CurrentTrack'] == track])
        print(f"      '{track}': {count} entries")

conn.close()

print("\n" + "="*80)
print("DIAGNOSIS:")
if len(sandown_df) > 0 and sandown_df['CurrentTrack'].iloc[0] in track_mapping:
    print("  Sandown races ARE in database and CAN be matched to TrackID")
    print("  Issue must be in feature extraction or prediction step")
elif len(sandown_df) > 0:
    print("  Sandown races ARE in database but CANNOT be matched to TrackID")
    print("  Track name mismatch issue!")
else:
    print("  Sandown races NOT in upcoming betting database or filtered out")
print("="*80)
