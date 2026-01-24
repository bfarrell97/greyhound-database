"""Test upcoming race predictions"""
from greyhound_ml_model import GreyhoundMLModel

# Load model
model = GreyhoundMLModel()
model.load_model()

# Test predictions on upcoming races for today
predictions = model.predict_upcoming_races('2025-12-08', confidence_threshold=0.8)

print(f"\n{'='*80}")
print(f"PREDICTIONS FOR UPCOMING RACES ON 2025-12-08")
print(f"{'='*80}\n")

if len(predictions) > 0:
    print(f"Found {len(predictions)} high-confidence predictions:\n")
    for idx, row in predictions.iterrows():
        odds_str = f"${row['CurrentOdds']:.2f}" if row['CurrentOdds'] is not None else "N/A"
        print(f"{row['GreyhoundName']:<30} R{row['RaceNumber']:<2} {row['CurrentTrack']:<20} "
              f"Win Prob: {row['WinProbability']*100:5.1f}%  Odds: {odds_str}")
else:
    print("No high-confidence predictions found.")
    print("\nThis could mean:")
    print("  - No upcoming races scraped for this date")
    print("  - All greyhounds are new (no historical data)")
    print("  - No predictions meet the 80% confidence threshold")
