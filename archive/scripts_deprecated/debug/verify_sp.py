"""
Verify Starting Price (SP) Data Quality
Checks overround (market percentage) for each race
Valid range: 90% - 130% (typical greyhound races are 105-120%)
"""

import sqlite3
import pandas as pd

DB_PATH = 'greyhound_racing.db'

conn = sqlite3.connect(DB_PATH)

# Get races with SP data
query = """
SELECT 
    rm.MeetingDate,
    t.TrackName,
    r.RaceNumber,
    r.RaceID,
    ge.StartingPrice,
    ge.GreyhoundID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.StartingPrice IS NOT NULL
  AND ge.Position NOT IN ('SCR', '')
  AND rm.MeetingDate >= '2024-01-01'
"""

df = pd.read_sql_query(query, conn)
conn.close()

print(f"Total entries with SP: {len(df):,}")

# Parse SP (handle $3.50 format)
def parse_sp(sp):
    if pd.isna(sp): return None
    try:
        return float(str(sp).replace('$', '').strip())
    except:
        return None

df['Odds'] = df['StartingPrice'].apply(parse_sp)
df = df[df['Odds'].notna() & (df['Odds'] > 1)]  # Valid odds > $1

# Calculate implied probability
df['ImpliedProb'] = 1 / df['Odds']

# Group by race and calculate overround
race_stats = df.groupby(['RaceID', 'MeetingDate', 'TrackName', 'RaceNumber']).agg({
    'ImpliedProb': 'sum',
    'GreyhoundID': 'count'
}).reset_index()
race_stats.columns = ['RaceID', 'MeetingDate', 'Track', 'Race', 'Overround', 'Runners']
race_stats['OverroundPct'] = race_stats['Overround'] * 100

print(f"\nTotal races analyzed: {len(race_stats):,}")

# Flag issues
low_overround = race_stats[race_stats['OverroundPct'] < 90]
high_overround = race_stats[race_stats['OverroundPct'] > 130]
valid = race_stats[(race_stats['OverroundPct'] >= 90) & (race_stats['OverroundPct'] <= 130)]

print(f"\n=== OVERROUND DISTRIBUTION ===")
print(f"Valid (90-130%):     {len(valid):,} races ({len(valid)/len(race_stats)*100:.1f}%)")
print(f"Too Low (<90%):      {len(low_overround):,} races ({len(low_overround)/len(race_stats)*100:.1f}%)")
print(f"Too High (>130%):    {len(high_overround):,} races ({len(high_overround)/len(race_stats)*100:.1f}%)")

print(f"\n=== STATISTICS ===")
print(f"Min Overround:  {race_stats['OverroundPct'].min():.1f}%")
print(f"Max Overround:  {race_stats['OverroundPct'].max():.1f}%")
print(f"Mean Overround: {race_stats['OverroundPct'].mean():.1f}%")
print(f"Median:         {race_stats['OverroundPct'].median():.1f}%")

# Show sample issues
if len(low_overround) > 0:
    print(f"\n=== SAMPLE LOW OVERROUND RACES (<90%) ===")
    print(low_overround.sort_values('OverroundPct').head(10).to_string(index=False))

if len(high_overround) > 0:
    print(f"\n=== SAMPLE HIGH OVERROUND RACES (>130%) ===")
    print(high_overround.sort_values('OverroundPct', ascending=False).head(10).to_string(index=False))

# Check for common issues
print(f"\n=== RUNNER COUNT DISTRIBUTION ===")
print(race_stats['Runners'].value_counts().sort_index().head(10))
