"""
Quick test of ML model with small sample size
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import sys

# Import the main model class
from greyhound_ml_model import GreyhoundMLModel

def main():
    """Test with small sample"""
    print("="*80)
    print("QUICK ML MODEL TEST (Small Sample)")
    print("="*80)

    model = GreyhoundMLModel()

    # Use just 1 month of training data and 1 week of test data
    print("\nExtracting TRAINING data (Dec 2024 only)...")
    train_df = model.extract_training_data(start_date='2024-12-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return

    print(f"Training samples: {len(train_df)}")

    # Test on first week of 2025
    print("\nExtracting TEST data (Jan 1-7, 2025)...")
    test_df = model.extract_training_data(start_date='2025-01-01', end_date='2025-01-07')

    if test_df is None or len(test_df) == 0:
        print("ERROR: Failed to extract test data")
        return

    print(f"Test samples: {len(test_df)}")

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, train_df_with_features = model.prepare_features(train_df)
    X_test, y_test, test_df_with_features = model.prepare_features(test_df)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Train model
    print("\nTraining model...")
    model.train_model(X_train, y_train, X_test, y_test)

    # Quick evaluation
    print("\nQuick Evaluation...")
    y_pred = model.model.predict(X_test)
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Test value betting logic with the small sample
    print("\n" + "="*80)
    print("VALUE BETTING TEST")
    print("="*80)

    test_df_with_pred = test_df_with_features.copy()
    test_df_with_pred['PredProba'] = y_pred_proba
    test_df_with_pred['IsWinner'] = y_test.values

    # Try 70% confidence
    conf_threshold = 0.70
    value_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= conf_threshold].copy()

    print(f"\nBets with >={conf_threshold*100:.0f}% confidence: {len(value_bets)}")

    if len(value_bets) > 0:
        # Calculate implied probability from odds
        def safe_convert_odds(x):
            if pd.notna(x) and x not in [0, '0', 'None', None]:
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 100
            return 100

        value_bets['ImpliedProb'] = 1 / value_bets['StartingPrice'].apply(safe_convert_odds)

        print(f"Sample starting prices: {value_bets['StartingPrice'].head(10).tolist()}")
        print(f"Sample implied probs: {value_bets['ImpliedProb'].head(10).tolist()}")

        # Filter to value bets
        value_bets = value_bets[value_bets['PredProba'] > value_bets['ImpliedProb']]

        print(f"Value bets (model prob > implied prob): {len(value_bets)}")

        if len(value_bets) > 0:
            value_wins = value_bets['IsWinner'].sum()
            print(f"Value bet winners: {value_wins}")
            print(f"Strike rate: {value_wins/len(value_bets)*100:.1f}%")

            # Calculate profit
            value_profit = 0
            for _, bet in value_bets.iterrows():
                if bet['IsWinner'] == 1:
                    try:
                        odds = float(bet['StartingPrice']) if pd.notna(bet['StartingPrice']) else 0
                        value_profit += (odds - 1)
                    except (ValueError, TypeError):
                        pass
                else:
                    value_profit -= 1

            print(f"Profit/Loss: ${value_profit:.2f}")
            print(f"ROI: {value_profit/len(value_bets)*100:.1f}%")
        else:
            print("No value bets found (all model probs <= implied probs)")
    else:
        print("No bets met confidence threshold")

    print("\n" + "="*80)
    print("TEST COMPLETE - Full model should work correctly!")
    print("="*80)

if __name__ == '__main__':
    main()
