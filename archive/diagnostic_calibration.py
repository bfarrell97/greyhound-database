"""Diagnostic to compare raw vs calibrated predictions"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import sys

# Load model
ml_model = GreyhoundMLModel()
ml_model.load_model()

print(f"Calibration enabled: {ml_model.use_calibration}")
print(f"Calibration parameters: a={ml_model.calibration_a}, b={ml_model.calibration_b}")

# Get backtest data
conn = sqlite3.connect('greyhound_racing.db')

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
WHERE rm.MeetingDate >= '2025-06-01'
  AND rm.MeetingDate <= '2025-11-30'
  AND ge.Position IS NOT NULL
  AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT LIKE '%TAS%'
ORDER BY rm.MeetingDate, r.RaceNumber, ge.Box
"""

race_df = pd.read_sql_query(query, conn)
print(f"\nLoaded {len(race_df)} race entries")

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
WHERE rm.MeetingDate < '2025-06-01'
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

hist_df = pd.read_sql_query(hist_query, conn)

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
WHERE rm.MeetingDate < '2025-06-01'
  AND ge.Position IS NOT NULL
  AND ge.Box IS NOT NULL
  AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT LIKE '%TAS%'
GROUP BY t.TrackID, r.Distance, ge.Box
"""

box_stats_df = pd.read_sql_query(box_stats_query, conn)
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
    if idx % 20000 == 0:
        print(f"  {idx}/{len(race_df)}")
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
print(f"Extracted {len(features_df)} feature vectors")

# Prepare features
X = features_df[ml_model.feature_columns]

# Get raw and calibrated predictions
raw_probs = ml_model.model.predict_proba(X)[:, 1]
calibrated_probs = ml_model.apply_calibration(raw_probs)

print(f"\n--- RAW PREDICTIONS ---")
print(f"  Min: {raw_probs.min():.6f}")
print(f"  Max: {raw_probs.max():.6f}")
print(f"  Mean: {raw_probs.mean():.6f}")
print(f"  Median: {np.median(raw_probs):.6f}")

print(f"\n--- CALIBRATED PREDICTIONS ---")
print(f"  Min: {calibrated_probs.min():.6f}")
print(f"  Max: {calibrated_probs.max():.6f}")
print(f"  Mean: {calibrated_probs.mean():.6f}")
print(f"  Median: {np.median(calibrated_probs):.6f}")

# Check calibration by binning
print(f"\n--- CALIBRATION BY RAW PREDICTION BINS ---")
for low in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = (raw_probs >= low) & (raw_probs < low + 0.1)
    if mask.sum() > 0:
        mean_raw = raw_probs[mask].mean()
        mean_calibrated = calibrated_probs[mask].mean()
        mean_actual = features_df.loc[mask, 'IsWinner'].mean()
        print(f"  Raw {low:.1f}-{low+0.1:.1f}: calibrated={mean_calibrated:.3f}, actual={mean_actual:.3f}")

# Check by starting price
print(f"\n--- STRIKE RATE BY ODDS BRACKET (calibrated) ---")
features_df['RawProb'] = raw_probs
features_df['CalibratedProb'] = calibrated_probs

for low_odds, high_odds in [(1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 20.0)]:
    mask = (features_df['StartingPrice'] >= low_odds) & (features_df['StartingPrice'] < high_odds)
    if mask.sum() > 0:
        count = mask.sum()
        wins = features_df.loc[mask, 'IsWinner'].sum()
        strike = wins / count * 100
        mean_prob = features_df.loc[mask, 'CalibratedProb'].mean() * 100
        print(f"  ${low_odds:.1f}-${high_odds:.1f}: {count:>6} bets, {wins:>5} wins, {strike:>5.1f}% strike, {mean_prob:>5.1f}% pred")
