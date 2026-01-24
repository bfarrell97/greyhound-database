"""
Backtest model on November 2025 data to check for overfitting

This script will:
1. Make predictions for each day in November 2025 (using data BEFORE that date)
2. Compare predictions to actual race outcomes
3. Calculate win rate, ROI, and other metrics
4. Determine if model is overfit or performing well on unseen data
"""

import sqlite3
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel
from datetime import datetime, timedelta
import pickle

print("="*80)
print("NOVEMBER 2025 BACKTEST - OVERFITTING VALIDATION")
print("="*80)

# Load the model
model = GreyhoundMLModel()
model.load_model()

# Get all November dates
conn = sqlite3.connect('greyhound_racing.db')

dates_query = """
SELECT DISTINCT rm.MeetingDate
FROM RaceMeetings rm
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate < '2025-12-01'
ORDER BY rm.MeetingDate
"""

nov_dates = pd.read_sql_query(dates_query, conn)['MeetingDate'].tolist()
print(f"\nTesting on {len(nov_dates)} days in November 2025\n")

# Results storage
all_predictions = []
all_actuals = []

# Test different confidence thresholds
thresholds = [0.5, 0.6, 0.7, 0.8]

for threshold in thresholds:
    print(f"\n{'='*80}")
    print(f"TESTING AT {threshold*100:.0f}% CONFIDENCE THRESHOLD")
    print(f"{'='*80}\n")

    total_predictions = 0
    correct_predictions = 0
    races_with_predictions = {}

    for date in nov_dates:
        # Get predictions for this date
        try:
            predictions = model.predict_upcoming_races(date, confidence_threshold=threshold)

            if len(predictions) == 0:
                continue

            # Get actual results for this date
            results_query = """
            SELECT
                g.GreyhoundName,
                t.TrackName,
                r.RaceNumber,
                ge.FinishPosition,
                ge.Box
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE rm.MeetingDate = ?
            AND ge.FinishPosition IS NOT NULL
            AND ge.FinishPosition != 'DNF'
            AND ge.FinishPosition != 'SCR'
            """

            results = pd.read_sql_query(results_query, conn, params=(date,))

            # Match predictions to actual outcomes
            for _, pred in predictions.iterrows():
                greyhound = pred['GreyhoundName']
                track = pred['CurrentTrack']
                race_num = pred['RaceNumber']

                # Find actual result
                actual = results[
                    (results['GreyhoundName'] == greyhound) &
                    (results['TrackName'] == track) &
                    (results['RaceNumber'] == race_num)
                ]

                if len(actual) > 0:
                    position = actual.iloc[0]['FinishPosition']

                    try:
                        position = int(position)
                        total_predictions += 1

                        # Did we predict a winner?
                        if position == 1:
                            correct_predictions += 1
                            result = "✓ WIN"
                        else:
                            result = f"✗ {position}"

                        # Store for detailed analysis
                        race_key = f"{date}_{track}_R{race_num}"
                        if race_key not in races_with_predictions:
                            races_with_predictions[race_key] = []

                        races_with_predictions[race_key].append({
                            'greyhound': greyhound,
                            'position': position,
                            'win_prob': pred['WinProbability'],
                            'result': result
                        })

                    except:
                        pass  # Skip non-numeric positions

        except Exception as e:
            print(f"Error processing {date}: {e}")
            continue

    # Calculate metrics
    if total_predictions > 0:
        win_rate = (correct_predictions / total_predictions) * 100
        baseline_random = (1 / 8) * 100  # Assuming 8-dog fields on average

        print(f"\nRESULTS:")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Win rate: {win_rate:.2f}%")
        print(f"  Random baseline: {baseline_random:.2f}%")
        print(f"  Improvement over random: {win_rate - baseline_random:+.2f}%")

        # Show some example predictions
        print(f"\n  Sample races with predictions:")
        for i, (race_key, preds) in enumerate(list(races_with_predictions.items())[:5]):
            print(f"\n  {race_key}:")
            for p in sorted(preds, key=lambda x: x['win_prob'], reverse=True):
                print(f"    {p['greyhound']:<25} Prob: {p['win_prob']*100:>5.1f}%  {p['result']}")

        # ROI calculation (simplified - assuming flat betting)
        # In real betting, you'd use odds and kelly criterion
        roi = (correct_predictions / total_predictions) - 1
        print(f"\n  Simplified ROI (no odds): {roi*100:+.1f}%")
        print(f"  (Note: Real ROI depends on odds at time of bet)")
    else:
        print(f"  No predictions made at this threshold")

conn.close()

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("""
To determine if model is overfit, compare these results to training performance:

✓ GOOD SIGNS (Not Overfit):
  - Win rate > 15% at 50% confidence
  - Win rate > 25% at 80% confidence
  - Win rate within 20% of training performance
  - Consistent performance across different confidence levels

❌ BAD SIGNS (Overfit):
  - Win rate < 15% at 50% confidence
  - Win rate close to random (12.5%)
  - Much worse than training performance (>30% drop)
  - High confidence predictions no better than low confidence

NEXT STEPS:
1. If validation passes → Run more detailed backtests with actual odds/ROI
2. If validation fails → Retrain with walk-forward optimization
3. Either way → DO NOT bet real money until 2+ weeks paper trading
""")
