"""Quick diagnostic to understand prediction distribution"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import sys

# Load model
ml_model = GreyhoundMLModel()
ml_model.load_model()

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
print(f"Loaded {len(hist_df)} historical races")

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
print("\nExtracting features...")
features_list = []
for idx, row in race_df.iterrows():
    if idx % 10000 == 0:
        print(f"  {idx}/{len(race_df)} ({idx/len(race_df)*100:.1f}%)")
        sys.stdout.flush()
    
    if row['GreyhoundID'] not in hist_grouped.groups:
        continue
    
    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
    
    last_5 = greyhound_hist.head(5)
    if len(last_5) < 5:
        continue
    
    # Build features
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
    
    # ADD BOOKMAKERPROB
    try:
        sp = float(row['StartingPrice']) if row['StartingPrice'] else 0
        features['BookmakerProb'] = 1.0 / sp if sp > 0 else 0.5
    except (ValueError, TypeError):
        features['BookmakerProb'] = 0.5
    
    features['EntryID'] = row['EntryID']
    features['GreyhoundName'] = row['GreyhoundName']
    features['StartingPrice'] = row['StartingPrice']
    features['IsWinner'] = (pd.to_numeric(row['Position'], errors='coerce') == 1)
    features_list.append(features)

conn.close()

features_df = pd.DataFrame(features_list)
print(f"\nExtracted {len(features_df)} feature vectors")

# Prepare features for prediction
X = features_df[ml_model.feature_columns]

# Make predictions
print("\nMaking predictions...")
probs = ml_model.model.predict_proba(X)[:, 1]

# Analyze distribution
print(f"\nPrediction Distribution:")
print(f"  Min: {probs.min():.6f}")
print(f"  Max: {probs.max():.6f}")
print(f"  Mean: {probs.mean():.6f}")
print(f"  Median: {np.median(probs):.6f}")
print(f"  Std Dev: {probs.std():.6f}")

print(f"\nPrediction Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(probs, p):.6f}")

print(f"\nBets at different confidence thresholds:")
for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    count = (probs >= threshold).sum()
    pct = 100 * count / len(probs)
    print(f"  >= {threshold*100:>5.0f}%: {count:>6,} bets ({pct:>5.1f}%)")

# Check a few high-probability predictions
print(f"\nTop 10 highest probability predictions:")
top_idx = np.argsort(probs)[-10:][::-1]
for i, idx in enumerate(top_idx, 1):
    dog = features_df.iloc[idx]['GreyhoundName']
    sp = features_df.iloc[idx]['StartingPrice']
    winner = features_df.iloc[idx]['IsWinner']
    prob = probs[idx]
    print(f"  {i}. {dog:20s} prob={prob:.4f} odds={sp:8} winner={winner}")
