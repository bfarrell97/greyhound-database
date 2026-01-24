"""
Backtest model on November 2025 HISTORICAL data

This directly uses the model's prediction logic on historical race data
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import pickle

print("="*80)
print("NOVEMBER 2025 HISTORICAL BACKTEST")
print("="*80)

# Load model using the model class
m = GreyhoundMLModel()
m.load_model()

print(f"\nModel loaded:")
print(f"  Features: {len(m.feature_columns)}")
print(f"  Model type: {type(m.model)}")

# Get November 2025 data
conn = sqlite3.connect('greyhound_racing.db')

# Get all greyhound entries from November 2025
query = """
SELECT
    ge.EntryID,
    g.GreyhoundName,
    t.TrackName,
    t.TrackKey,
    rm.MeetingDate,
    r.RaceNumber,
    r.Distance,
    ge.Box,
    ge.Weight,
    ge.Position,
    ge.FinishTime,
    tr.TrainerName,
    ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate < '2025-12-01'
AND ge.Position IS NOT NULL
AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

print("\nLoading November 2025 data...")
all_entries = pd.read_sql_query(query, conn)
print(f"  Total entries: {len(all_entries):,}")

# Convert Position to numeric, mark winners
all_entries['PositionNum'] = pd.to_numeric(all_entries['Position'], errors='coerce')
all_entries['Won'] = (all_entries['PositionNum'] == 1).astype(int)

print(f"  Winners: {all_entries['Won'].sum():,}")
print(f"  Random baseline win rate: {all_entries['Won'].mean()*100:.2f}%")

# Now extract features for each entry (same way model does it)
print("\nExtracting features from historical data...")
print("  (This mimics what the model does for predictions)")

# We'll use the same feature extraction logic from the model
# Get all historical races for context (excluding November 2025)
m.load_historical_races()

# Extract features for these entries
features_list = []
entry_info = []

for _, entry in all_entries.iterrows():
    try:
        features = m.extract_features_for_greyhound(
            greyhound_name=entry['GreyhoundName'],
            track_name=entry['TrackName'],
            distance=entry['Distance'],
            box=entry['Box'],
            current_date=entry['MeetingDate']
        )

        if features is not None:
            features_list.append(features)
            entry_info.append({
                'EntryID': entry['EntryID'],
                'GreyhoundName': entry['GreyhoundName'],
                'TrackName': entry['TrackName'],
                'MeetingDate': entry['MeetingDate'],
                'RaceNumber': entry['RaceNumber'],
                'Box': entry['Box'],
                'Won': entry['Won'],
                'Position': entry['PositionNum'],
                'StartingPrice': entry['StartingPrice']
            })
    except Exception as e:
        pass  # Skip entries we can't extract features for

print(f"  Successfully extracted features for {len(features_list):,} entries")

if len(features_list) == 0:
    print("\nERROR: Could not extract any features. Model cannot be backtested.")
    print("This suggests a problem with feature extraction logic.")
    conn.close()
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

    # Random baseline
    avg_field_size = all_entries.groupby(['MeetingDate', 'TrackName', 'RaceNumber']).size().mean()
    random_baseline = (1 / avg_field_size) * 100

    print(f"\n  Total predictions: {total:,}")
    print(f"  Actual winners: {winners}")
    print(f"  Win rate: {win_rate:.2f}%")
    print(f"  Random baseline: {random_baseline:.2f}%")
    print(f"  Improvement: {win_rate - random_baseline:+.2f}%")

    #Simplified ROI (assuming $1 bet on each at average odds)
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
            print(f"    ROI: {roi:+.2f}%")

    # Show sample predictions
    print(f"\n  Sample predictions (top 10 by probability):")
    samples = predictions.nlargest(10, 'WinProbability')
    for _, row in samples.iterrows():
        result = "WIN" if row['Won'] == 1 else f"P{int(row['Position'])}"
        odds_str = f"${row['StartingPrice']:.2f}" if pd.notna(row['StartingPrice']) else "N/A"
        print(f"    {row['GreyhoundName']:<25} {row['TrackName']:<15} R{int(row['RaceNumber']):>2}  "
              f"Prob: {row['WinProbability']*100:>5.1f}%  Odds: {odds_str:>6}  {result}")

conn.close()

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
1. If results are good -> Proceed to live paper trading
2. If results are bad -> Model needs retraining with walk-forward
3. Either way -> Track live performance for 2+ weeks before betting real money
""")
