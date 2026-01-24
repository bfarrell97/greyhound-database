"""
Extended test of promising track weight configurations
Tests both 80% and 90% confidence levels with more data
"""
import sys
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel
from sklearn.metrics import roc_auc_score

def test_weight_config_extended(metro_weight, provincial_weight, country_weight):
    """Test a specific track weight configuration with extended metrics"""
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

    # Use 2 months for better sample size
    print("\nExtracting TRAINING data (Nov-Dec 2024)...")
    train_df = model.extract_training_data(start_date='2024-11-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return None

    print(f"Training samples: {len(train_df)}")

    # Test data (Jan 2025)
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
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Value betting test at multiple confidence levels
    test_df_with_pred = test_df_with_features.copy()
    test_df_with_pred['PredProba'] = y_pred_proba
    test_df_with_pred['IsWinner'] = y_test.values

    results = {
        'metro': metro_weight,
        'provincial': provincial_weight,
        'country': country_weight,
        'roc_auc': roc_auc
    }

    # Test both 80% and 90% confidence
    for conf_threshold in [0.80, 0.90]:
        value_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= conf_threshold].copy()

        if len(value_bets) > 0:
            def safe_convert_odds(x):
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
                avg_odds = value_bets['StartingPrice'].apply(safe_convert_odds).mean()
            else:
                roi = 0
                strike_rate = 0
                avg_odds = 0
                value_bets = pd.DataFrame()
        else:
            roi = 0
            strike_rate = 0
            avg_odds = 0

        conf_str = f"{int(conf_threshold*100)}pct"
        results[f'num_bets_{conf_str}'] = len(value_bets)
        results[f'strike_rate_{conf_str}'] = strike_rate
        results[f'roi_{conf_str}'] = roi
        results[f'avg_odds_{conf_str}'] = avg_odds

    print(f"\nRESULTS:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  80% Conf: {results['num_bets_80pct']} bets, {results['strike_rate_80pct']:.1f}% strike, {results['roi_80pct']:.1f}% ROI, avg odds {results['avg_odds_80pct']:.2f}")
    print(f"  90% Conf: {results['num_bets_90pct']} bets, {results['strike_rate_90pct']:.1f}% strike, {results['roi_90pct']:.1f}% ROI, avg odds {results['avg_odds_90pct']:.2f}")

    return results

def main():
    """Test promising configurations from initial sweep"""
    print("="*80)
    print("EXTENDED TRACK WEIGHT TESTING - PROMISING CONFIGS")
    print("="*80)

    # Focus on configs with positive ROI from initial test
    # Plus a few variations to explore
    configs = [
        (1.0, 0.7, 0.5),   # Best ROI in initial test (36.8%)
        (2.0, 1.5, 0.5),   # Second best ROI (18.6%)
        (1.5, 1.0, 0.3),   # Third best ROI (13.1%)
        (1.0, 1.0, 1.0),   # Baseline - no weighting
        # Additional variations to explore
        (1.0, 0.8, 0.5),   # Between best and 1.0/1.0/1.0
        (1.0, 0.6, 0.5),   # Less provincial weight
        (1.0, 0.7, 0.4),   # Less country weight
        (1.0, 0.7, 0.3),   # Even less country weight
        (2.0, 1.5, 0.3),   # Combine good metro boost with less country
    ]

    results = []
    for metro, provincial, country in configs:
        result = test_weight_config_extended(metro, provincial, country)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - 80% CONFIDENCE")
    print("="*80)
    print(f"{'Metro':<8} {'Prov':<8} {'Country':<8} {'ROC-AUC':<10} {'Bets':<8} {'Strike%':<10} {'ROI%':<10} {'AvgOdds':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['metro']:<8.1f} {r['provincial']:<8.1f} {r['country']:<8.1f} "
              f"{r['roc_auc']:<10.4f} {r['num_bets_80pct']:<8} {r['strike_rate_80pct']:<10.1f} "
              f"{r['roi_80pct']:<10.1f} {r['avg_odds_80pct']:<10.2f}")

    print("\n" + "="*80)
    print("SUMMARY - 90% CONFIDENCE")
    print("="*80)
    print(f"{'Metro':<8} {'Prov':<8} {'Country':<8} {'ROC-AUC':<10} {'Bets':<8} {'Strike%':<10} {'ROI%':<10} {'AvgOdds':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['metro']:<8.1f} {r['provincial']:<8.1f} {r['country']:<8.1f} "
              f"{r['roc_auc']:<10.4f} {r['num_bets_90pct']:<8} {r['strike_rate_90pct']:<10.1f} "
              f"{r['roi_90pct']:<10.1f} {r['avg_odds_90pct']:<10.2f}")

    # Find best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    # Best ROI at 80%
    best_roi_80 = max(results, key=lambda x: x['roi_80pct'])
    print(f"\nBEST ROI (80% confidence):")
    print(f"  Metro={best_roi_80['metro']}, Provincial={best_roi_80['provincial']}, Country={best_roi_80['country']}")
    print(f"  ROI: {best_roi_80['roi_80pct']:.1f}%, Bets: {best_roi_80['num_bets_80pct']}, Strike: {best_roi_80['strike_rate_80pct']:.1f}%")

    # Best ROI at 90%
    best_roi_90 = max(results, key=lambda x: x['roi_90pct'])
    print(f"\nBEST ROI (90% confidence):")
    print(f"  Metro={best_roi_90['metro']}, Provincial={best_roi_90['provincial']}, Country={best_roi_90['country']}")
    print(f"  ROI: {best_roi_90['roi_90pct']:.1f}%, Bets: {best_roi_90['num_bets_90pct']}, Strike: {best_roi_90['strike_rate_90pct']:.1f}%")

    # Best ROC-AUC
    best_auc = max(results, key=lambda x: x['roc_auc'])
    print(f"\nBEST ROC-AUC:")
    print(f"  Metro={best_auc['metro']}, Provincial={best_auc['provincial']}, Country={best_auc['country']}")
    print(f"  ROC-AUC: {best_auc['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
