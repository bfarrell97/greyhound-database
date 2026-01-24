"""Check which greyhounds have odds vs predictions"""
import sqlite3
from greyhound_ml_model import GreyhoundMLModel

# Get predictions
model = GreyhoundMLModel()
model.load_model()
predictions = model.predict_upcoming_races('2025-12-08', 0.5)

print(f"\nPredictions: {len(predictions)} greyhounds")
predicted_names = set(predictions['GreyhoundName'].tolist())

# Get greyhounds with odds
conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT GreyhoundName FROM UpcomingBettingRunners WHERE CurrentOdds IS NOT NULL AND CurrentOdds > 0')
names_with_odds = set([row[0] for row in cursor.fetchall()])
print(f"Greyhounds with odds in DB: {len(names_with_odds)}")

# Check overlap
overlap = predicted_names.intersection(names_with_odds)
print(f"\nGreyhounds with BOTH predictions AND odds: {len(overlap)}")

if len(overlap) > 0:
    print("\nSample greyhounds with both:")
    for name in list(overlap)[:10]:
        cursor.execute('SELECT CurrentOdds FROM UpcomingBettingRunners WHERE GreyhoundName = ?', (name,))
        result = cursor.fetchone()
        odds = result[0] if result else None
        prob = predictions[predictions['GreyhoundName'] == name]['WinProbability'].values[0]
        if odds:
            print(f"  {name:<30} ${odds:.2f}  Prob: {prob*100:.1f}%")
        else:
            print(f"  {name:<30} NO ODDS   Prob: {prob*100:.1f}%")
else:
    print("\nNo greyhounds have both predictions and odds!")
    print("\nSample predicted greyhounds (no odds):")
    for name in list(predicted_names)[:5]:
        print(f"  {name}")
    print("\nSample greyhounds with odds (no predictions):")
    for name in list(names_with_odds)[:5]:
        print(f"  {name}")

conn.close()
