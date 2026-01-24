"""
OPTIMIZED November 2025 backtest using BULK processing like the model
This should process 29,650 entries in minutes, not hours
"""
import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

print("="*80)
print("NOVEMBER 2025 BACKTEST - BULK PROCESSING VERSION")
print("="*80)

# Load model
m = GreyhoundMLModel()
m.load_model()

print(f"\nModel loaded:")
print(f"  Features: {len(m.feature_columns)}")

# Connect to database
conn = sqlite3.connect('greyhound_racing.db')

# Get all November 2025 races
print("\nLoading November 2025 race data...")
query = """
SELECT
    ge.EntryID,
    ge.GreyhoundID,
    g.GreyhoundName,
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
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate < '2025-12-01'
AND ge.Position IS NOT NULL
AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

df = pd.read_sql_query(query, conn)
print(f"  Total entries: {len(df):,}")

# Filter out NZ and Tasmania tracks
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded_tracks)]
# Also filter out tracks with (NZ) or NZ in the name (case insensitive)
df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]
# Filter out Tasmania tracks
df = df[~df['TrackName'].str.contains('TAS', na=False, case=False)]
print(f"  After filtering NZ/TAS tracks: {len(df):,}")

# Convert Position to numeric, mark winners
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)

# Convert StartingPrice to numeric
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')

print(f"  Winners: {df['IsWinner'].sum():,}")
print(f"  Random baseline: {df['IsWinner'].mean()*100:.2f}%")

# BULK LOAD: Get ALL historical races before November 2025 in ONE query
print("\nBulk loading historical data (this is the key optimization)...")
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
    WHERE rm.MeetingDate < '2025-11-01'
      AND ge.Position IS NOT NULL
      AND ge.SplitBenchmarkLengths IS NOT NULL
      AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
      AND ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
    ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
"""

hist_df = pd.read_sql_query(historical_query, conn)
print(f"  Loaded {len(hist_df):,} historical races")

# Calculate box win rates (using data BEFORE November 2025)
print("\nCalculating box win rates...")
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
    WHERE rm.MeetingDate < '2025-11-01'
      AND ge.Position IS NOT NULL
      AND ge.Box IS NOT NULL
    GROUP BY t.TrackID, r.Distance, ge.Box
"""
box_stats_df = pd.read_sql_query(box_stats_query, conn)
box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']

# Create lookup dict
box_win_rates = {}
for _, row in box_stats_df.iterrows():
    key = (row['TrackID'], row['Distance'], row['Box'])
    box_win_rates[key] = row['BoxWinRate']

print(f"  Box win rates calculated for {len(box_win_rates)} combinations")

conn.close()

# Group historical data by greyhound for fast lookup
print("\nGrouping historical data by greyhound...")
hist_grouped = hist_df.groupby('GreyhoundID')
print(f"  Grouped into {len(hist_grouped)} greyhounds")

# Extract features efficiently (same approach as model training)
print("\nExtracting features from historical data...")
features_list = []
entry_info = []
total = len(df)
skipped = 0

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  Processed {idx}/{total} entries ({idx/total*100:.1f}%)")

    # Get this greyhound's history before this race
    if row['GreyhoundID'] not in hist_grouped.groups:
        skipped += 1
        continue

    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    # Filter to races before current date
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]

    # Take last 5 races
    last_5 = greyhound_hist.head(5)

    if len(last_5) < 5:
        skipped += 1
        continue

    # Build features (same logic as model)
    features = {}

    # Box win rate
    box_key = (row['TrackID'], row['Distance'], row['Box'])
    features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)

    # Recent form from last 3 races
    last_3 = last_5.head(3)
    if len(last_3) > 0:
        last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
        features['AvgPositionLast3'] = last_3_positions.mean()
        features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
    else:
        features['AvgPositionLast3'] = 4.5
        features['WinRateLast3'] = 0

    # GM_OT_ADJ features with track tier weighting
    for i, (_, race) in enumerate(last_5.iterrows(), 1):
        gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)

        # Apply track tier weight
        track_weight = m.get_track_tier_weight(race['TrackName'])
        features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight

    features_list.append(features)
    entry_info.append({
        'EntryID': row['EntryID'],
        'GreyhoundName': row['GreyhoundName'],
        'TrackName': row['TrackName'],
        'MeetingDate': row['MeetingDate'],
        'RaceNumber': row['RaceNumber'],
        'Box': row['Box'],
        'Won': row['IsWinner'],
        'Position': row['Position'],
        'StartingPrice': row['StartingPrice']
    })

print(f"  Successfully extracted features for {len(features_list):,} entries")
print(f"  Skipped {skipped:,} entries (insufficient historical data)")

if len(features_list) == 0:
    print("\nERROR: Could not extract any features. Cannot backtest.")
    exit(1)

# Create feature DataFrame
X = pd.DataFrame(features_list)
entry_df = pd.DataFrame(entry_info)

# Ensure we have all required feature columns
missing_cols = set(m.feature_columns) - set(X.columns)
for col in missing_cols:
    X[col] = 0

X = X[m.feature_columns]

# Make predictions
print("\nMaking predictions...")
probabilities = m.model.predict_proba(X)[:, 1]
entry_df['WinProbability'] = probabilities

print(f"  Predictions made: {len(entry_df):,}")
print(f"  Probability range: {probabilities.min():.3f} to {probabilities.max():.3f}")

# Test at different confidence thresholds
thresholds = [0.5, 0.6, 0.7, 0.8]

print("\n" + "="*80)
print("BACKTEST RESULTS")
print("="*80)

for threshold in thresholds:
    print(f"\n{'='*80}")
    print(f"CONFIDENCE THRESHOLD: {threshold*100:.0f}%")
    print(f"{'='*80}")

    # Filter to predictions above threshold
    predictions = entry_df[entry_df['WinProbability'] >= threshold].copy()

    if len(predictions) == 0:
        print(f"  No predictions at this threshold")
        continue

    # Calculate metrics
    total = len(predictions)
    winners = predictions['Won'].sum()
    win_rate = (winners / total) * 100

    # Random baseline (average field size)
    avg_field_size = df.groupby(['MeetingDate', 'TrackName', 'RaceNumber']).size().mean()
    random_baseline = (1 / avg_field_size) * 100

    print(f"\n  Total predictions: {total:,}")
    print(f"  Actual winners: {winners}")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Random baseline: {random_baseline:.2f}%")
    print(f"  Improvement: {win_rate - random_baseline:+.2f}%")

    # ROI calculation (assuming $1 bet on each at starting price)
    if 'StartingPrice' in predictions.columns:
        bets_with_odds = predictions[predictions['StartingPrice'].notna()].copy()
        if len(bets_with_odds) > 0:
            total_stake = len(bets_with_odds)
            winners_with_odds = bets_with_odds[bets_with_odds['Won'] == 1]
            total_return = (winners_with_odds['StartingPrice'].sum() if len(winners_with_odds) > 0 else 0)
            roi = ((total_return - total_stake) / total_stake) * 100

            print(f"\n  ROI CALCULATION (using starting prices):")
            print(f"    Bets placed: {total_stake}")
            print(f"    Winners with odds: {len(winners_with_odds)}")
            print(f"    Total staked: ${total_stake:.2f}")
            print(f"    Total return: ${total_return:.2f}")
            print(f"    Profit/Loss: ${total_return - total_stake:+.2f}")
            print(f"    ROI: {roi:+.2f}%")

    # Show sample predictions
    print(f"\n  Sample predictions (top 10 by probability):")
    samples = predictions.nlargest(10, 'WinProbability')
    for _, row in samples.iterrows():
        result = "WIN" if row['Won'] == 1 else f"P{int(row['Position'])}"
        odds_str = f"${row['StartingPrice']:.2f}" if pd.notna(row['StartingPrice']) else "N/A"
        print(f"    {row['GreyhoundName']:<25} {row['TrackName']:<15} R{int(row['RaceNumber']):>2}  "
              f"Prob: {row['WinProbability']*100:>5.1f}%  Odds: {odds_str:>6}  {result}")

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("""
INTERPRETATION:

GOOD SIGNS (Not Overfit):
  - Win rate > 15% at 50% confidence
  - Win rate > 25% at 80% confidence
  - Positive ROI at higher confidence levels
  - Win rate improves as confidence increases

BAD SIGNS (Overfit):
  - Win rate close to random baseline (~12.5%)
  - Negative ROI even at high confidence
  - Win rate doesn't improve with higher confidence
  - Very few predictions at high confidence levels

NEXT STEPS:
1. If results are GOOD -> Proceed to live paper trading for 2+ weeks
2. If results are BAD -> Model needs retraining with walk-forward validation
3. Either way -> DO NOT bet real money without paper trading validation
""")
