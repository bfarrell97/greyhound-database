"""
Test different track tier weighting configurations to find optimal weights
"""
import sys
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel

def test_weight_config(metro_weight, provincial_weight, country_weight):
    """Test a specific track weight configuration"""
    print("\n" + "="*80)
    print(f"TESTING: Metro={metro_weight}, Provincial={provincial_weight}, Country={country_weight}")
    print("="*80)

    # Initialize model with custom weights
    model = GreyhoundMLModel()
    model.TRACK_WEIGHTS = {
        'metro': metro_weight,
        'provincial': provincial_weight,
        'country': country_weight
    }

    # Use small dataset for quick testing
    print("\nExtracting TRAINING data (Dec 2024)...")
    train_df = model.extract_training_data(start_date='2024-12-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return None

    print(f"Training samples: {len(train_df)}")

    # Test data
    print("\nExtracting TEST data (Jan 2025)...")
    test_df = model.extract_training_data(start_date='2025-01-01', end_date='2025-01-31')

    if test_df is None or len(test_df) == 0:
        print("ERROR: Failed to extract test data")
        return None

    print(f"Test samples: {len(test_df)}")

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, train_df_with_features = model.prepare_features(train_df)
    X_test, y_test, test_df_with_features = model.prepare_features(test_df)

    # Train model
    print("\nTraining model...")
    model.train_model(X_train, y_train, X_test, y_test)

    # Quick evaluation
    from sklearn.metrics import roc_auc_score
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Value betting test at 80% confidence
    test_df_with_pred = test_df_with_features.copy()
    test_df_with_pred['PredProba'] = y_pred_proba
    test_df_with_pred['IsWinner'] = y_test.values

    # 80% confidence value bets
    value_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= 0.80].copy()

    if len(value_bets) > 0:
        def safe_convert_odds(x):
            import pandas as pd
            if pd.notna(x) and x not in [0, '0', 'None', None]:
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 100
            return 100

        value_bets['ImpliedProb'] = 1 / value_bets['StartingPrice'].apply(safe_convert_odds)
        value_bets = value_bets[value_bets['PredProba'] > value_bets['ImpliedProb']]

        if len(value_bets) > 0:
            value_wins = value_bets['IsWinner'].sum()

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

            roi = value_profit / len(value_bets) * 100
            strike_rate = value_wins / len(value_bets) * 100
        else:
            roi = 0
            strike_rate = 0
            value_bets = []
    else:
        roi = 0
        strike_rate = 0

    print(f"\nRESULTS:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Value bets (80%): {len(value_bets)}")
    print(f"  Strike rate: {strike_rate:.1f}%")
    print(f"  ROI: {roi:.1f}%")

    return {
        'metro': metro_weight,
        'provincial': provincial_weight,
        'country': country_weight,
        'roc_auc': roc_auc,
        'num_bets': len(value_bets),
        'strike_rate': strike_rate,
        'roi': roi
    }

def main():
    """Test multiple weight configurations"""
    print("="*80)
    print("TRACK WEIGHT PARAMETER SWEEP")
    print("="*80)

    # Test configurations
    # Format: (metro, provincial, country)
    configs = [
        (1.0, 0.7, 0.5),   # Original
        (1.5, 1.0, 0.5),   # Current
        (2.0, 1.0, 0.5),   # More metro emphasis
        (1.5, 1.0, 0.3),   # Penalize country more
        (2.0, 1.5, 0.5),   # Boost both metro and provincial
        (2.0, 1.0, 0.3),   # Max differential
        (1.0, 1.0, 1.0),   # Baseline (no weighting)
    ]

    results = []
    for metro, provincial, country in configs:
        result = test_weight_config(metro, provincial, country)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*80)
    print(f"{'Metro':<8} {'Prov':<8} {'Country':<8} {'ROC-AUC':<10} {'Bets':<8} {'Strike%':<10} {'ROI%':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['metro']:<8.1f} {r['provincial']:<8.1f} {r['country']:<8.1f} "
              f"{r['roc_auc']:<10.4f} {r['num_bets']:<8} {r['strike_rate']:<10.1f} {r['roi']:<10.1f}")

    # Find best by ROI
    best_roi = max(results, key=lambda x: x['roi'])
    print(f"\nBEST ROI: Metro={best_roi['metro']}, Provincial={best_roi['provincial']}, "
          f"Country={best_roi['country']} => ROI={best_roi['roi']:.1f}%")

    # Find best by ROC-AUC
    best_auc = max(results, key=lambda x: x['roc_auc'])
    print(f"BEST ROC-AUC: Metro={best_auc['metro']}, Provincial={best_auc['provincial']}, "
          f"Country={best_auc['country']} => ROC-AUC={best_auc['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
