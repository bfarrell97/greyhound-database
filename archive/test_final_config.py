"""
Full test of (1.0, 0.7, 0.3) track weights with odds bracket analysis
"""
import sys
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel
from sklearn.metrics import roc_auc_score

def main():
    """Test the (1.0, 0.7, 0.3) configuration with full dataset"""
    print("="*80)
    print("FULL MODEL TEST - Track Weights: Metro=1.0, Provincial=0.7, Country=0.3")
    print("="*80)

    # Initialize model with specified weights
    model = GreyhoundMLModel()
    model.TRACK_WEIGHTS = {
        'metro': 1.0,
        'provincial': 0.7,
        'country': 0.3
    }

    # Extract training data (2023-2024)
    print("\nExtracting TRAINING data (2023-2024)...")
    train_df = model.extract_training_data(start_date='2023-01-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return

    print(f"Training samples: {len(train_df)}")

    # Extract test data (2025 H1)
    print("\nExtracting TEST data (2025 H1)...")
    test_df = model.extract_training_data(start_date='2025-01-01', end_date='2025-06-30')

    if test_df is None or len(test_df) == 0:
        print("ERROR: Failed to extract test data")
        return

    print(f"Test samples: {len(test_df)}")

    # Prepare features
    print("\nPreparing training features...")
    X_train, y_train, train_df_with_features = model.prepare_features(train_df)

    print("Preparing test features...")
    X_test, y_test, test_df_with_features = model.prepare_features(test_df)

    # Train model
    print("\nTraining model...")
    model.train_model(X_train, y_train, X_test, y_test)

    # Evaluation
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC: {roc_auc:.4f}")

    # Value betting analysis at 80% and 85% confidence
    test_df_with_pred = test_df_with_features.copy()
    test_df_with_pred['PredProba'] = y_pred_proba
    test_df_with_pred['IsWinner'] = y_test.values

    def safe_convert_odds(x):
        if pd.notna(x) and x not in [0, '0', 'None', None]:
            try:
                return float(x)
            except (ValueError, TypeError):
                return 100
        return 100

    # Test both 80% and 85% confidence
    for conf_threshold in [0.80, 0.85]:
        print("\n" + "="*80)
        print(f"VALUE BETTING ANALYSIS - {int(conf_threshold*100)}% CONFIDENCE")
        print("="*80)

        value_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= conf_threshold].copy()

        if len(value_bets) == 0:
            print(f"No bets meet {int(conf_threshold*100)}% confidence threshold")
            continue

        value_bets['ImpliedProb'] = 1 / value_bets['StartingPrice'].apply(safe_convert_odds)
        value_bets = value_bets[value_bets['PredProba'] > value_bets['ImpliedProb']]

        if len(value_bets) == 0:
            print("No value bets found (all model probs <= implied probs)")
            continue

        # Add odds column for bracket analysis
        value_bets['Odds'] = value_bets['StartingPrice'].apply(safe_convert_odds)

        # FILTER: Remove bets under $1.50 odds (consistently losing)
        value_bets_filtered = value_bets[value_bets['Odds'] >= 1.50].copy()

        print(f"\nBEFORE ODDS FILTER (>=1.50):")
        print(f"  Total bets: {len(value_bets)}")

        # Overall stats BEFORE filter
        value_wins_before = value_bets['IsWinner'].sum()
        value_profit_before = 0
        for _, bet in value_bets.iterrows():
            if bet['IsWinner'] == 1:
                value_profit_before += (bet['Odds'] - 1)
            else:
                value_profit_before -= 1

        print(f"  Wins: {value_wins_before}")
        print(f"  Strike rate: {value_wins_before/len(value_bets)*100:.1f}%")
        print(f"  Profit/Loss: ${value_profit_before:.2f}")
        print(f"  ROI: {value_profit_before/len(value_bets)*100:.1f}%")

        # Overall stats AFTER filter
        value_wins = value_bets_filtered['IsWinner'].sum()
        value_profit = 0
        for _, bet in value_bets_filtered.iterrows():
            if bet['IsWinner'] == 1:
                value_profit += (bet['Odds'] - 1)
            else:
                value_profit -= 1

        print(f"\nAFTER ODDS FILTER (>=1.50):")
        print(f"  Total value bets: {len(value_bets_filtered)}")
        print(f"  Wins: {value_wins}")
        print(f"  Losses: {len(value_bets_filtered) - value_wins}")
        print(f"  Strike rate: {value_wins/len(value_bets_filtered)*100:.1f}%")
        print(f"  Profit/Loss: ${value_profit:.2f}")
        print(f"  ROI: {value_profit/len(value_bets_filtered)*100:.1f}%")
        print(f"  Avg odds: {value_bets_filtered['Odds'].mean():.2f}")

        # ODDS BRACKET ANALYSIS (50c increments)
        print(f"\n" + "-"*80)
        print("ODDS BRACKET ANALYSIS (50c increments)")
        print("-"*80)
        print(f"{'Odds Range':<15} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'P/L':<10} {'ROI%':<10}")
        print("-"*80)

        # Define brackets: 1.01-1.50, 1.51-2.00, 2.01-2.50, etc.
        brackets = []
        start = 1.01
        while start <= 20.0:  # Go up to odds of 20
            end = start + 0.49
            brackets.append((start, end))
            start = end + 0.01

        # Add a final bracket for very high odds
        brackets.append((20.01, 999.99))

        for start_odds, end_odds in brackets:
            bracket_bets = value_bets_filtered[(value_bets_filtered['Odds'] >= start_odds) & (value_bets_filtered['Odds'] <= end_odds)]

            if len(bracket_bets) == 0:
                continue

            bracket_wins = bracket_bets['IsWinner'].sum()
            bracket_profit = 0
            for _, bet in bracket_bets.iterrows():
                if bet['IsWinner'] == 1:
                    bracket_profit += (bet['Odds'] - 1)
                else:
                    bracket_profit -= 1

            strike_rate = bracket_wins / len(bracket_bets) * 100
            roi = bracket_profit / len(bracket_bets) * 100

            if end_odds > 999:
                odds_range = f"{start_odds:.2f}+"
            else:
                odds_range = f"{start_odds:.2f}-{end_odds:.2f}"

            print(f"{odds_range:<15} {len(bracket_bets):<8} {bracket_wins:<8} {strike_rate:<10.1f} "
                  f"${bracket_profit:<9.2f} {roi:<10.1f}")

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)

    feature_importance = pd.DataFrame({
        'feature': model.feature_columns,
        'importance': model.model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # Save the model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    model.save_model()
    print("Model saved to greyhound_model.pkl")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
