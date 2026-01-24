"""
Test different odds ranges to find optimal betting window
Tests: $1.50-$2.00, $1.50-$2.50, $1.50-$3.00 on 2025 data
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'
CONFIDENCE_THRESHOLD = 0.80
INITIAL_BANKROLL = 1000.0

print("="*80)
print("TESTING ODDS RANGES: $1.50-$3.00 on 2025 data")
print("="*80)

# Load trained model
print("\nLoading trained ML model...")
ml_model = GreyhoundMLModel()
try:
    ml_model.load_model()
    print(f"Model loaded successfully")
    print(f"  Features: {ml_model.feature_columns}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Load all race entries
print(f"\nLoading race data from {START_DATE} to {END_DATE}...")
query = """
SELECT
    ge.EntryID,
    g.GreyhoundName,
    g.GreyhoundID,
    t.TrackName,
    t.TrackID,
    rm.MeetingDate,
    r.RaceNumber,
    r.Distance,
    ge.Box,
    ge.Weight,
    ge.Position,
    ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
print(f"Loaded {len(df):,} race entries")

# Filter out NZ and Tasmania tracks
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded_tracks)]
df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]
print(f"After filtering NZ/TAS tracks: {len(df):,}")

# Convert Position to numeric, mark winners
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce').fillna(2.0)
df['StartingPrice'] = df['StartingPrice'].clip(lower=1.5)

print(f"Winners: {df['IsWinner'].sum():,}")

# BULK LOAD historical data
print("\nBulk loading historical data...")
historical_query = """
    SELECT
        ge.GreyhoundID,
        t.TrackName,
        r.Distance,
        ge.Weight,
        ge.Position,
        ge.FinishTime,
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

hist_df = pd.read_sql_query(historical_query, conn, params=(START_DATE,))
print(f"Loaded {len(hist_df):,} historical races")

# Calculate box win rates
print("Calculating box win rates...")
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

# Create lookup dict
box_win_rates = {}
for _, row in box_stats_df.iterrows():
    key = (row['TrackID'], row['Distance'], row['Box'])
    box_win_rates[key] = row['BoxWinRate']

# Group historical data
hist_grouped = hist_df.groupby('GreyhoundID')

# Extract features
print("Extracting features...")
features_list = []
for idx, row in df.iterrows():
    if idx % 50000 == 0 and idx > 0:
        print(f"  Processed {idx:,}/{len(df):,} entries...")
    
    if row['GreyhoundID'] not in hist_grouped.groups:
        continue
    
    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
    last_5 = greyhound_hist.head(5)
    
    if len(last_5) < 5:
        continue
    
    features = {}
    features['EntryID'] = row['EntryID']
    features['GreyhoundName'] = row['GreyhoundName']
    features['TrackName'] = row['TrackName']
    features['MeetingDate'] = row['MeetingDate']
    features['RaceNumber'] = row['RaceNumber']
    features['StartingPrice'] = row['StartingPrice']
    features['IsWinner'] = row['IsWinner']
    
    # Extract model features
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
    
    # GM_OT_ADJ with baseline recency
    recency_weights = [2.0, 1.5, 1.0, 1.0, 1.0]
    for i, (_, race) in enumerate(last_5.iterrows(), 1):
        gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
        track_weight = ml_model.get_track_tier_weight(race['TrackName'])
        recency_weight = recency_weights[i - 1]
        features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight * recency_weight
    
    features_list.append(features)

print(f"Successfully extracted features for {len(features_list):,} entries")
features_df = pd.DataFrame(features_list)

# Make predictions
print("\nMaking predictions...")
X = features_df[ml_model.feature_columns]
raw_predictions = ml_model.model.predict_proba(X)[:, 1]

# DON'T apply calibration - use raw predictions (backtest_staking_strategies.py doesn't use it)
features_df['Probability'] = raw_predictions
features_df['ImpliedProb'] = 1 / features_df['StartingPrice']
features_df['Value'] = features_df['Probability'] > features_df['ImpliedProb']

# Filter by confidence
high_conf = features_df[features_df['Probability'] >= CONFIDENCE_THRESHOLD].copy()
value_bets = high_conf[high_conf['Value']].copy()

print(f"High confidence (>={CONFIDENCE_THRESHOLD:.0%}): {len(high_conf):,}")
print(f"With value edge: {len(value_bets):,}")

# Test different odds ranges
print("\n" + "="*80)
print("TESTING ODDS RANGES WITH FLAT 2% STAKING")
print("="*80)

odds_ranges = [
    (1.50, 2.00, "$1.50-$2.00 (Original)"),
    (1.50, 2.50, "$1.50-$2.50"),
    (1.50, 3.00, "$1.50-$3.00 (Expanded)"),
    (2.00, 3.00, "$2.00-$3.00 (Mid-range only)"),
]

results = []

for low_odds, high_odds, label in odds_ranges:
    bracket = value_bets[(value_bets['StartingPrice'] >= low_odds) & (value_bets['StartingPrice'] < high_odds)].copy()
    
    if len(bracket) == 0:
        print(f"\n{label}")
        print(f"  No bets found")
        continue
    
    # Flat 2% staking
    stake_pct = 0.02
    total_staked = len(bracket) * INITIAL_BANKROLL * stake_pct
    bracket['Stake'] = INITIAL_BANKROLL * stake_pct
    bracket['Return'] = np.where(bracket['IsWinner'] == 1, bracket['StartingPrice'] * bracket['Stake'], 0)
    bracket['PnL'] = bracket['Return'] - bracket['Stake']
    
    wins = bracket['IsWinner'].sum()
    total_pnl = bracket['PnL'].sum()
    roi = (total_pnl / total_staked) * 100
    strike = (wins / len(bracket)) * 100
    avg_odds = bracket['StartingPrice'].mean()
    
    results.append({
        'Range': label,
        'Bets': len(bracket),
        'Wins': wins,
        'Strike%': strike,
        'AvgOdds': avg_odds,
        'Staked': total_staked,
        'PnL': total_pnl,
        'ROI%': roi,
    })
    
    print(f"\n{label}")
    print(f"  Bets: {len(bracket):>4}  Wins: {wins:>3}  Strike: {strike:>5.1f}%")
    print(f"  Avg Odds: {avg_odds:.2f}  Staked: ${total_staked:>8.2f}")
    print(f"  P/L: ${total_pnl:>8.2f}  ROI: {roi:>6.2f}%")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Range':<20} {'Bets':>6} {'Wins':>5} {'Strike%':>8} {'Avg Odds':>9} {'ROI%':>8}")
print("-"*80)
for r in results:
    print(f"{r['Range']:<20} {r['Bets']:>6.0f} {r['Wins']:>5.0f} {r['Strike%']:>7.1f}% {r['AvgOdds']:>9.2f} {r['ROI%']:>7.2f}%")

conn.close()
