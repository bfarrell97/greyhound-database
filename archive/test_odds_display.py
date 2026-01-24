"""Test if CurrentOdds is in predictions"""
from greyhound_ml_model import GreyhoundMLModel
import pandas as pd

model = GreyhoundMLModel()
model.load_model()

predictions = model.predict_upcoming_races('2025-12-08', 0.5)

print(f"\n{'='*80}")
print(f"PREDICTIONS CHECK")
print(f"{'='*80}")
print(f"\nTotal predictions: {len(predictions)}")
print(f"\nColumns in predictions DataFrame:")
print(predictions.columns.tolist())

if len(predictions) > 0:
    print(f"\nFirst 5 predictions:")
    print(predictions.head())

    print(f"\nCurrentOdds column info:")
    print(f"  Null count: {predictions['CurrentOdds'].isna().sum()}")
    print(f"  Non-null count: {predictions['CurrentOdds'].notna().sum()}")
    print(f"\nSample CurrentOdds values:")
    print(predictions[['GreyhoundName', 'CurrentOdds']].head(10))
