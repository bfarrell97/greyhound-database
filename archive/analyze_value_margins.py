"""
Analyze value margins: What's the actual edge model is claiming?
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'

print("="*80)
print("VALUE MARGIN ANALYSIS: What edge is model claiming?")
print("="*80)

ml_model = GreyhoundMLModel()
ml_model.load_model()

conn = sqlite3.connect(DB_PATH)

# Load test data
print("\nLoading race data...")
query = """
SELECT
    ge.EntryID, g.GreyhoundID, t.TrackID, r.Distance, ge.Box, ge.Position, ge.StartingPrice, rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND ge.StartingPrice >= 1.5
  AND ge.StartingPrice < 2.0
ORDER BY rm.MeetingDate
"""

df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')

# Filter NZ/TAS
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df.index.map(lambda x: any(str(df.loc[x, 'GreyhoundID']).startswith(t) for t in excluded_tracks))]

print(f"Loaded {len(df):,} races in $1.50-$2.00 range")

# Load historical data for features
print("Loading historical data...")
hist_query = """
    SELECT ge.GreyhoundID, t.TrackName, r.Distance, ge.Weight, ge.Position,
           ge.SplitBenchmarkLengths as G_Split_ADJ, rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
           ge.FinishTimeBenchmarkLengths as G_OT_ADJ, rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
           rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < ? AND ge.Position IS NOT NULL
      AND ge.SplitBenchmarkLengths IS NOT NULL
      AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
      AND ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT LIKE '%TAS%'
    ORDER BY ge.GreyhoundID, rm.MeetingDate DESC
"""

hist_df = pd.read_sql_query(hist_query, conn, params=(START_DATE,))

# Box win rates
box_stats_query = """
    SELECT t.TrackID, r.Distance, ge.Box, COUNT(*) as TotalRaces,
           SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < ? AND ge.Position IS NOT NULL AND ge.Box IS NOT NULL
    GROUP BY t.TrackID, r.Distance, ge.Box
"""
box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(START_DATE,))
box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']
box_win_rates = {(row['TrackID'], row['Distance'], row['Box']): row['BoxWinRate'] 
                 for _, row in box_stats_df.iterrows()}

# Extract features (simpler version)
print("Extracting features...")
hist_grouped = hist_df.groupby('GreyhoundID')
features_list = []

for idx, row in df.iterrows():
    if idx % 25000 == 0 and idx > 0:
        print(f"  {idx:,}/{len(df):,}")
    
    if row['GreyhoundID'] not in hist_grouped.groups:
        continue
    
    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
    last_5 = greyhound_hist.head(5)
    
    if len(last_5) < 5:
        continue
    
    features = {}
    features['EntryID'] = row['EntryID']
    features['IsWinner'] = row['IsWinner']
    features['StartingPrice'] = row['StartingPrice']
    
    box_key = (row['TrackID'], row['Distance'], row['Box'])
    features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)
    
    last_3 = last_5.head(3)
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    features['AvgPositionLast3'] = last_3_positions.mean() if len(last_3) > 0 else 4.5
    features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3) if len(last_3) > 0 else 0
    
    recency_weights = [2.0, 1.5, 1.0, 1.0, 1.0]
    for i, (_, race) in enumerate(last_5.iterrows(), 1):
        gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
        track_weight = ml_model.get_track_tier_weight(race['TrackName'])
        recency_weight = recency_weights[i - 1]
        features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight * recency_weight
    
    features_list.append(features)

features_df = pd.DataFrame(features_list)
print(f"Extracted {len(features_df):,} features")

# Make predictions
print("\nMaking predictions...")
X = features_df[ml_model.feature_columns]
raw_predictions = ml_model.model.predict_proba(X)[:, 1]
features_df['PredictedProb'] = raw_predictions
features_df['ImpliedProb'] = 1 / features_df['StartingPrice']
features_df['Margin'] = features_df['PredictedProb'] - features_df['ImpliedProb']
features_df['MarginPct'] = (features_df['Margin'] / features_df['ImpliedProb']) * 100

# Analyze by confidence bins
print("\n" + "="*80)
print("VALUE MARGIN BY CONFIDENCE LEVEL ($1.50-$2.00)")
print("="*80)

confidence_bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]

results = []

for low_conf, high_conf in confidence_bins:
    bin_data = features_df[(features_df['PredictedProb'] >= low_conf) & (features_df['PredictedProb'] < high_conf)]
    
    if len(bin_data) == 0:
        continue
    
    wins = bin_data['IsWinner'].sum()
    win_rate = (wins / len(bin_data)) * 100
    avg_margin = bin_data['Margin'].mean()
    avg_margin_pct = bin_data['MarginPct'].mean()
    implied_prob = bin_data['ImpliedProb'].mean()
    
    results.append({
        'Confidence': f"{low_conf:.0%}-{high_conf:.0%}",
        'Bets': len(bin_data),
        'PredProb%': (low_conf + high_conf) / 2 * 100,
        'ActualWin%': win_rate,
        'ImpliedProb%': implied_prob * 100,
        'AvgMargin': avg_margin * 100,
        'AvgMarginPct': avg_margin_pct,
    })
    
    print(f"\n{low_conf:.0%}-{high_conf:.0%} confidence ({len(bin_data)} bets):")
    print(f"  Predicted avg prob: {(low_conf + high_conf) / 2:.1%}")
    print(f"  Actual win rate: {win_rate:.1f}%")
    print(f"  Implied prob (from odds): {implied_prob:.1%}")
    print(f"  Avg margin: {avg_margin:.4f} ({avg_margin_pct:.1f}%)")
    print(f"  Win difference: {win_rate - (low_conf + high_conf) / 2 * 100:+.1f}%")

# Value threshold analysis
print("\n" + "="*80)
print("IMPACT OF VALUE THRESHOLD (Minimum margin required)")
print("="*80)

margin_thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]

for margin_threshold in margin_thresholds:
    value_bets = features_df[features_df['Margin'] >= margin_threshold]
    
    if len(value_bets) == 0:
        continue
    
    wins = value_bets['IsWinner'].sum()
    win_rate = (wins / len(value_bets)) * 100
    
    # Calculate ROI with 2% flat staking
    stake_pct = 0.02
    bankroll = 1000.0
    total_staked = len(value_bets) * bankroll * stake_pct
    returns = (value_bets[value_bets['IsWinner'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
    pnl = returns - total_staked
    roi = (pnl / total_staked) * 100
    
    print(f"Margin >= {margin_threshold:+.1%}: {len(value_bets):>4} bets, {win_rate:>5.1f}% strike, ROI {roi:>6.2f}%")

conn.close()

# Summary
print("\n" + "="*80)
print("FINDINGS")
print("="*80)
print("""
The value filter (predicted > implied) is too loose. Almost all high-confidence bets
pass this filter because the model is significantly overconfident.

KEY INSIGHT: We need a MINIMUM MARGIN REQUIREMENT to ensure we're only betting
when the model has a meaningful edge beyond just beating the implied probability.

RECOMMENDATION: Test margin thresholds (e.g., require model to be 5-10% better 
than implied odds) to find optimal edge vs bet volume tradeoff.
""")
