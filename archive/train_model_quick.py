"""
Quick model training with recent data only for fast testing
"""
from greyhound_ml_model import GreyhoundMLModel

def main():
    print("="*80)
    print("QUICK MODEL TRAINING (Nov-Dec 2024 training, Jan 2025 testing)")
    print("="*80)

    model = GreyhoundMLModel()

    # Use recent data for faster training
    print("\nExtracting TRAINING data (Nov-Dec 2024)...")
    train_df = model.extract_training_data(start_date='2024-11-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return

    print(f"Training samples: {len(train_df)}")

    # Test data
    print("\nExtracting TEST data (Jan 2025)...")
    test_df = model.extract_training_data(start_date='2025-01-01', end_date='2025-01-31')

    if test_df is None or len(test_df) == 0:
        print("ERROR: Failed to extract test data")
        return

    print(f"Test samples: {len(test_df)}")

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, train_df_with_features = model.prepare_features(train_df)
    X_test, y_test, test_df_with_features = model.prepare_features(test_df)

    # Train model
    print("\nTraining model...")
    model.train_model(X_train, y_train, X_test, y_test)

    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate_model(X_train, y_train, X_test, y_test, test_df_with_features)

    # Save model
    model.save_model()

    print("\n" + "="*80)
    print("QUICK TRAINING COMPLETE - Model saved to greyhound_model.pkl")
    print("="*80)
    print("\nYou can now use the GUI to load predictions.")
    print("The model includes:")
    print("  - Track tier weighting (Metro=1.0, Provincial=0.7, Country=0.3)")
    print("  - Value bet filter (model prob > implied prob from odds)")
    print("  - Odds filter (>= $1.50)")

if __name__ == '__main__':
    main()
