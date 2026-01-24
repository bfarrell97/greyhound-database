"""
PROPER November 2025 backtest using the model's actual feature extraction method
"""
import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

print("="*80)
print("NOVEMBER 2025 BACKTEST - OVERFITTING VALIDATION")
print("="*80)

# Load model
m = GreyhoundMLModel()
m.load_model()

print(f"\nModel loaded:")
print(f"  Features: {len(m.feature_columns)}")
print(f"  Feature columns: {m.feature_columns}")

# Get November 2025 data
conn = sqlite3.connect('greyhound_racing.db')

# Get all greyhound entries from November 2025
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

# Calculate box win rates (using data BEFORE November 2025)
print("\nCalculating box win rates from historical data...")
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

# Extract features for each entry using the model's actual method
print("\nExtracting features using model's _extract_greyhound_features method...")
features_list = []
entry_info = []
skipped = 0

for idx, entry in all_entries.iterrows():
    if idx % 1000 == 0:
        print(f"  Processed {idx}/{len(all_entries)} entries ({idx/len(all_entries)*100:.1f}%)")

    try:
        # Use the model's actual feature extraction method
        features = m._extract_greyhound_features(
            conn=conn,
            greyhound_id=entry['GreyhoundID'],
            current_date=entry['MeetingDate'],
            current_track=entry['TrackName'],
            current_distance=entry['Distance'],
            box=entry['Box'],
            weight=entry['Weight'],
            track_id=entry['TrackID'],
            prize_money=None,
            box_win_rates=box_win_rates
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
        else:
            skipped += 1
    except Exception as e:
        skipped += 1

conn.close()

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

    # Random baseline
    avg_field_size = all_entries.groupby(['MeetingDate', 'TrackName', 'RaceNumber']).size().mean()
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
