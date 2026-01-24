"""
Overfitting diagnostic script

Common signs of overfitting:
1. Training accuracy >> Test accuracy (model memorized training data)
2. Model performs well on training period but poorly on unseen future data
3. Very high confidence predictions that don't match market odds
4. Model picks favorites that are already heavily backed in the market
"""

import sqlite3
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel
import pickle

print("="*80)
print("OVERFITTING DIAGNOSTIC REPORT")
print("="*80)

# Load the current model
model_data = pickle.load(open('greyhound_model.pkl', 'rb'))

print("\n1. MODEL COMPLEXITY CHECK")
print("-" * 40)
if hasattr(model_data['model'], 'get_params'):
    params = model_data['model'].get_params()
    print(f"Max depth: {params.get('max_depth', 'N/A')}")
    print(f"N estimators: {params.get('n_estimators', 'N/A')}")
    print(f"Learning rate: {params.get('learning_rate', 'N/A')}")
    print(f"Min child weight: {params.get('min_child_weight', 'N/A')}")

    # Check for overfitting indicators
    max_depth = params.get('max_depth', 0)
    if max_depth and max_depth > 10:
        print("⚠️  WARNING: max_depth > 10 may cause overfitting")
    min_child_weight = params.get('min_child_weight', 1)
    if min_child_weight and min_child_weight < 1:
        print("⚠️  WARNING: min_child_weight < 1 may cause overfitting")

print("\n2. PREDICTION DISTRIBUTION CHECK")
print("-" * 40)
print("Loading model and making predictions...")
m = GreyhoundMLModel()
m.load_model()

# Get predictions for today
predictions = m.predict_upcoming_races('2025-12-08', confidence_threshold=0.0)  # Get ALL predictions

if len(predictions) > 0:
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"\nWin probability distribution:")
    print(predictions['WinProbability'].describe())

    # Check if model is just picking favorites
    print(f"\nHigh confidence (>80%): {(predictions['WinProbability'] > 0.8).sum()}")
    print(f"Medium confidence (50-80%): {((predictions['WinProbability'] >= 0.5) & (predictions['WinProbability'] <= 0.8)).sum()}")
    print(f"Low confidence (<50%): {(predictions['WinProbability'] < 0.5).sum()}")

    # Compare model probabilities to market odds
    print("\n3. MODEL vs MARKET COMPARISON")
    print("-" * 40)

    with_odds = predictions[predictions['CurrentOdds'].notna()].copy()
    if len(with_odds) > 0:
        print(f"\nPredictions with market odds: {len(with_odds)}")

        # Calculate implied probability from odds
        with_odds['ImpliedProb'] = 1.0 / with_odds['CurrentOdds']
        with_odds['ModelProb'] = with_odds['WinProbability']
        with_odds['Difference'] = with_odds['ModelProb'] - with_odds['ImpliedProb']

        print(f"\nModel probability vs Market probability:")
        print(f"  Correlation: {with_odds['ModelProb'].corr(with_odds['ImpliedProb']):.3f}")
        print(f"  (Correlation near 1.0 = model just copying market)")
        print(f"  (Correlation near 0 = model ignoring market)")

        # Check if model is just picking heavily backed favorites
        favorites = with_odds[with_odds['CurrentOdds'] <= 3.0]  # Odds <= $3
        print(f"\n  Market favorites (odds <= $3.00): {len(favorites)}")
        if len(favorites) > 0:
            print(f"  Avg model prob for favorites: {favorites['ModelProb'].mean():.3f}")
            print(f"  Avg implied prob for favorites: {favorites['ImpliedProb'].mean():.3f}")

        # Check value bets
        value_bets = with_odds[with_odds['Difference'] > 0.10]  # Model > Market by 10%
        print(f"\n  Value bets (model >> market): {len(value_bets)}")
        if len(value_bets) > 0:
            print(f"  Average edge on value bets: {(value_bets['Difference'] / value_bets['ImpliedProb'] * 100).mean():.1f}%")
            print(f"\n  Top 5 value bets:")
            top_value = value_bets.nlargest(5, 'Difference')[['GreyhoundName', 'CurrentTrack', 'CurrentOdds', 'ModelProb', 'ImpliedProb']]
            for _, row in top_value.iterrows():
                edge = (row['ModelProb'] - row['ImpliedProb']) / row['ImpliedProb'] * 100
                print(f"    {row['GreyhoundName']:<25} {row['CurrentTrack']:<15} ${row['CurrentOdds']:>5.2f}  Model: {row['ModelProb']*100:>4.1f}%  Market: {row['ImpliedProb']*100:>4.1f}%  Edge: {edge:>+5.1f}%")
    else:
        print("No predictions with odds available")

print("\n4. HISTORICAL VALIDATION CHECK")
print("-" * 40)
print("Checking if we have historical results to validate against...")

conn = sqlite3.connect('greyhound_racing.db')

# Check recent results (within last 30 days)
recent_results = pd.read_sql_query("""
    SELECT COUNT(*) as count,
           MIN(MeetingDate) as earliest,
           MAX(MeetingDate) as latest
    FROM HistoricalRaces
    WHERE MeetingDate >= date('now', '-30 days')
""", conn)

print(f"\nRecent historical results (last 30 days):")
print(f"  Count: {recent_results['count'].values[0]}")
print(f"  Date range: {recent_results['earliest'].values[0]} to {recent_results['latest'].values[0]}")

if recent_results['count'].values[0] > 0:
    print("\n✓ You have recent data to validate predictions against!")
    print("  Suggestion: Run backtest on last 7 days to check real-world performance")
else:
    print("\n⚠️  WARNING: No recent historical data to validate against")
    print("  Your model is trained on old data (2023-2024)")
    print("  Without recent validation, you can't detect overfitting!")

conn.close()

print("\n" + "="*80)
print("OVERFITTING RISK ASSESSMENT")
print("="*80)

risk_factors = []

# Check various risk factors
if len(predictions) > 0:
    high_conf_pct = (predictions['WinProbability'] > 0.8).sum() / len(predictions) * 100
    if high_conf_pct > 10:
        risk_factors.append(f"High confidence predictions: {high_conf_pct:.1f}% (>10% suggests overconfidence)")

    if len(with_odds) > 0:
        correlation = with_odds['ModelProb'].corr(with_odds['ImpliedProb'])
        if correlation > 0.95:
            risk_factors.append(f"Model-market correlation: {correlation:.3f} (>0.95 suggests model is just copying market)")
        elif correlation < 0.3:
            risk_factors.append(f"Model-market correlation: {correlation:.3f} (<0.3 suggests model ignoring market entirely)")

if recent_results['count'].values[0] == 0:
    risk_factors.append("No recent validation data - can't verify model performance on unseen data")

if len(risk_factors) > 0:
    print("\n⚠️  RISK FACTORS DETECTED:")
    for i, risk in enumerate(risk_factors, 1):
        print(f"  {i}. {risk}")
else:
    print("\n✓ No obvious overfitting indicators detected")
    print("  However, you should still validate with live betting results")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
1. VALIDATE WITH RECENT DATA
   - Scrape last 7 days of results
   - Run predictions on past races
   - Compare predictions to actual outcomes

2. TRACK LIVE PERFORMANCE
   - Keep a log of all bets placed
   - Record: predicted prob, odds, edge, actual outcome
   - Calculate ROI weekly

3. REDUCE OVERFITTING RISK
   - Use lower max_depth (6-8 instead of 10+)
   - Increase min_child_weight (3-5)
   - Add more regularization (higher reg_alpha, reg_lambda)
   - Use cross-validation during training

4. AVOID CURVE FITTING
   - Don't retrain model daily on latest results
   - Only retrain monthly with significant new data
   - Keep a holdout test set that model NEVER sees

5. PAPER TRADE FIRST
   - Track predictions for 2 weeks without betting real money
   - Verify positive ROI before risking capital
""")
