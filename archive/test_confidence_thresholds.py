"""
Test different confidence thresholds to find optimal ROI
Tests 80%, 85%, 90%, 95% thresholds on $1.50-$3.00 odds
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import sys

# Load model
ml_model = GreyhoundMLModel()
ml_model.load_model()

# Configuration
DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'

# Load race data
conn = sqlite3.connect(DB_PATH)

query = """
SELECT DISTINCT
    ge.EntryID,
    ge.GreyhoundID,
    g.GreyhoundName,
    rm.MeetingDate,
    t.TrackName as CurrentTrack,
    t.TrackID,
    r.RaceNumber,
    r.Distance,
    ge.Box,
    ge.Weight,
    ge.StartingPrice,
    ge.Position
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ?
  AND rm.MeetingDate <= ?
  AND ge.Position IS NOT NULL
  AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT LIKE '%TAS%'
ORDER BY rm.MeetingDate, r.RaceNumber, ge.Box
"""

race_df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
print(f"Loaded {len(race_df)} race entries")

# Get historical data
hist_query = """
SELECT
    ge.GreyhoundID,
    t.TrackName,
    r.Distance,
    ge.Weight,
    ge.Position,
    ge.SplitBenchmarkLengths as G_Split_ADJ,
    rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
    ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
    rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.SplitBenchmarkLengths IS NOT NULL
  AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
  AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT LIKE '%TAS%'
ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
"""

hist_df = pd.read_sql_query(hist_query, conn, params=(START_DATE,))

# Get box stats
box_stats_query = """
SELECT
    t.TrackID,
    r.Distance,
    ge.Box,
    COUNT(*) as TotalRaces,
    SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Box IS NOT NULL
  AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT LIKE '%TAS%'
GROUP BY t.TrackID, r.Distance, ge.Box
"""

box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(START_DATE,))
box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']

box_win_rates = {}
for _, row in box_stats_df.iterrows():
    key = (row['TrackID'], row['Distance'], row['Box'])
    box_win_rates[key] = row['BoxWinRate']

hist_grouped = hist_df.groupby('GreyhoundID')

# Extract features
print("Extracting features...")
features_list = []
for idx, row in race_df.iterrows():
    if idx % 30000 == 0:
        print(f"  {idx}/{len(race_df)} ({idx/len(race_df)*100:.1f}%)")
        sys.stdout.flush()
    
    if row['GreyhoundID'] not in hist_grouped.groups:
        continue
    
    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
    
    last_5 = greyhound_hist.head(5)
    if len(last_5) < 5:
        continue
    
    features = {}
    box_key = (row['TrackID'], row['Distance'], row['Box'])
    features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)
    
    last_3 = last_5.head(3)
    if len(last_3) > 0:
        last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
        features['AvgPositionLast3'] = last_3_positions.mean()
        features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
    else:
        features['AvgPositionLast3'] = 4.5
        features['WinRateLast3'] = 0
    
    for i, (_, race) in enumerate(last_5.iterrows(), 1):
        gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
        track_weight = ml_model.get_track_tier_weight(race['TrackName'])
        features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight
    
    features['IsWinner'] = (pd.to_numeric(row['Position'], errors='coerce') == 1)
    features['StartingPrice'] = row['StartingPrice']
    features_list.append(features)

conn.close()

features_df = pd.DataFrame(features_list)
print(f"Extracted {len(features_df)} feature vectors\n")

# Prepare features
X = features_df[ml_model.feature_columns]

# Get predictions
raw_probs = ml_model.model.predict_proba(X)[:, 1]

# Filter to $1.50-$3.00 odds range
features_df['StartingPrice_numeric'] = pd.to_numeric(features_df['StartingPrice'], errors='coerce')
target_mask = (features_df['StartingPrice_numeric'] >= 1.5) & (features_df['StartingPrice_numeric'] <= 3.0)
target_df = features_df[target_mask].copy()
target_probs = raw_probs[target_mask]

print(f"Dogs in $1.50-$3.00 range: {len(target_df)}")
print(f"\nTesting confidence thresholds on $1.50-$3.00 odds:\n")

# Test different thresholds
thresholds = [0.80, 0.85, 0.90, 0.95]

for threshold in thresholds:
    mask = target_probs >= threshold
    if mask.sum() == 0:
        print(f"Threshold {threshold*100:.0f}%: 0 bets found")
        continue
    
    subset = target_df[mask]
    probs = target_probs[mask]
    
    # Calculate metrics
    bets = len(subset)
    wins = subset['IsWinner'].sum()
    strike = wins / bets * 100
    
    # Calculate ROI
    odds_numeric = pd.to_numeric(subset['StartingPrice'], errors='coerce')
    avg_odds = odds_numeric.mean()
    break_even = 1 / avg_odds * 100
    
    total_staked = bets * 20  # $20 per bet
    total_return = (odds_numeric[subset['IsWinner'] == 1] * 20).sum()
    profit = total_return - total_staked
    roi = (profit / total_staked) * 100
    
    print(f"Threshold {threshold*100:.0f}%:")
    print(f"  Bets: {bets}")
    print(f"  Wins: {wins}")
    print(f"  Strike Rate: {strike:.1f}% (break-even: {break_even:.1f}%)")
    print(f"  Avg Odds: {avg_odds:.2f}")
    print(f"  Profit/Loss: ${profit:.2f}")
    print(f"  ROI: {roi:.2f}%")
    print()
