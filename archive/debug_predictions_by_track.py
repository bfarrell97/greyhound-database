"""Debug why only some tracks get predictions"""
from greyhound_ml_model import GreyhoundMLModel
import pandas as pd

# Load model
model = GreyhoundMLModel()
model.load_model()

# Test predictions with detailed debug
print("\n" + "="*80)
print("DEBUGGING PREDICTIONS BY TRACK")
print("="*80)

# Run predictions at low confidence to see all possible predictions
predictions = model.predict_upcoming_races('2025-12-08', confidence_threshold=0.01)

if len(predictions) > 0:
    print(f"\nTotal predictions at 1% confidence: {len(predictions)}")

    # Group by track
    by_track = predictions.groupby('CurrentTrack').size().reset_index(name='count')
    print("\nPredictions by track:")
    print("-" * 60)
    for _, row in by_track.iterrows():
        print(f"  {row['CurrentTrack']:<30} {row['count']:>3} predictions")

    # Show confidence distribution
    print("\nConfidence distribution:")
    print("-" * 60)
    print(f"  >= 80%: {len(predictions[predictions['WinProbability'] >= 0.8])}")
    print(f"  >= 70%: {len(predictions[predictions['WinProbability'] >= 0.7])}")
    print(f"  >= 60%: {len(predictions[predictions['WinProbability'] >= 0.6])}")
    print(f"  >= 50%: {len(predictions[predictions['WinProbability'] >= 0.5])}")

    # Show top predictions per track
    print("\nTop prediction per track (sorted by confidence):")
    print("-" * 80)
    top_per_track = predictions.sort_values('WinProbability', ascending=False).groupby('CurrentTrack').first()
    for track, row in top_per_track.iterrows():
        print(f"  {track:<20} R{row['RaceNumber']:<2} {row['GreyhoundName']:<25} {row['WinProbability']*100:>5.1f}%")

else:
    print("\nNo predictions generated at all!")
