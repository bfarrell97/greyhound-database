"""
Investigate why model works on $1.50-$2.00 but fails elsewhere
Look for patterns in the winning vs losing bets
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'

print("="*80)
print("INVESTIGATING $1.50-$2.00 EDGE")
print("="*80)

ml_model = GreyhoundMLModel()
ml_model.load_model()

conn = sqlite3.connect(DB_PATH)

# Load race data with more info
print("\nLoading race data...")
query = """
SELECT
    ge.EntryID, g.GreyhoundID, g.GreyhoundName, t.TrackName, t.TrackID, r.Distance, 
    r.RaceNumber, ge.Box, ge.Weight, ge.Position, ge.StartingPrice, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY rm.MeetingDate
"""

df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')

# Filter NZ/TAS
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded_tracks)]
df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]

print(f"Loaded {len(df):,} races")

# Split into odds ranges
low_odds = df[(df['StartingPrice'] >= 1.50) & (df['StartingPrice'] < 2.00)]
mid_odds = df[(df['StartingPrice'] >= 2.00) & (df['StartingPrice'] < 3.00)]
high_odds = df[(df['StartingPrice'] >= 3.00)]

print(f"\nOdds ranges:")
print(f"  $1.50-$2.00: {len(low_odds):,} races, {low_odds['IsWinner'].sum():,} winners ({low_odds['IsWinner'].mean()*100:.1f}%)")
print(f"  $2.00-$3.00: {len(mid_odds):,} races, {mid_odds['IsWinner'].sum():,} winners ({mid_odds['IsWinner'].mean()*100:.1f}%)")
print(f"  $3.00+:      {len(high_odds):,} races, {high_odds['IsWinner'].sum():,} winners ({high_odds['IsWinner'].mean()*100:.1f}%)")

# Analyze by dog attributes
print("\n" + "="*80)
print("ANALYZING DOG ATTRIBUTES")
print("="*80)

for bracket_name, bracket_df in [("$1.50-$2.00", low_odds), ("$2.00-$3.00", mid_odds), ("$3.00+", high_odds)]:
    print(f"\n{bracket_name}:")
    
    # By weight
    bracket_df_copy = bracket_df.copy()
    bracket_df_copy['WeightCategory'] = pd.cut(bracket_df_copy['Weight'], bins=[0, 30, 33, 100], labels=['Light(<30)', 'Medium(30-33)', 'Heavy(33+)'])
    
    for cat in ['Light(<30)', 'Medium(30-33)', 'Heavy(33+)']:
        cat_data = bracket_df_copy[bracket_df_copy['WeightCategory'] == cat]
        if len(cat_data) > 0:
            wr = cat_data['IsWinner'].mean() * 100
            print(f"  {cat}: {len(cat_data):>5} races, win%: {wr:>5.1f}%")
    
    # By box
    print(f"  By box position:")
    for box in range(1, 9):
        box_data = bracket_df[bracket_df['Box'] == box]
        if len(box_data) > 0:
            wr = box_data['IsWinner'].mean() * 100
            print(f"    Box {box}: {len(box_data):>5} races, win%: {wr:>5.1f}%")
    
    # By distance
    print(f"  By distance (top 5):")
    dist_summary = bracket_df.groupby('Distance').agg({
        'IsWinner': ['count', 'sum']
    }).reset_index()
    dist_summary.columns = ['Distance', 'Count', 'Wins']
    dist_summary['WinRate'] = (dist_summary['Wins'] / dist_summary['Count'] * 100)
    dist_summary = dist_summary.sort_values('Count', ascending=False).head(5)
    for _, row in dist_summary.iterrows():
        print(f"    {int(row['Distance'])}m: {int(row['Count']):>5} races, win%: {row['WinRate']:>5.1f}%")

# Now look at track-specific performance
print("\n" + "="*80)
print("TRACK-SPECIFIC PERFORMANCE ($1.50-$2.00)")
print("="*80)

track_summary = low_odds.groupby('TrackName').agg({
    'IsWinner': ['count', 'sum'],
    'StartingPrice': 'mean'
}).reset_index()
track_summary.columns = ['Track', 'Count', 'Wins', 'AvgOdds']
track_summary['WinRate'] = (track_summary['Wins'] / track_summary['Count'] * 100)
track_summary = track_summary[track_summary['Count'] >= 20].sort_values('WinRate', ascending=False)

print(f"\nTracks with 20+ races ($1.50-$2.00), sorted by win rate:")
print(f"{'Track':<20} {'Races':>6} {'Wins':>5} {'Win%':>7} {'AvgOdds':>8}")
print("-"*80)
for _, row in track_summary.head(20).iterrows():
    print(f"{row['Track']:<20} {int(row['Count']):>6} {int(row['Wins']):>5} {row['WinRate']:>6.1f}% {row['AvgOdds']:>8.2f}")

conn.close()

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("""
The model's 63.7% win rate on $1.50-$2.00 suggests:
1. Bookmakers are slightly underpricing favorites in this range
2. Or the model is picking up on something real about which favorites are better

The key is: WHERE IS THIS EDGE? Is it:
- Specific tracks where the model does better?
- Specific dog weights/boxes?
- Specific race distances?
- Specific time periods (seasonality)?

This analysis will help determine if the edge is real or just variance.
""")
