"""
Train the greyhound racing model with new BookmakerProb feature
Run this script to retrain after making changes to features
"""

from greyhound_ml_model import GreyhoundMLModel
from sklearn.model_selection import train_test_split

print("="*80)
print("TRAINING GREYHOUND RACING MODEL WITH BOOKMAKERPROB FEATURE")
print("="*80)

# Create model instance
model = GreyhoundMLModel()

# Extract training data (2023-01-01 to 2025-05-31, before June 2025 backtest period)
print("\n1. Extracting training data from 2023-2025...")
df = model.extract_training_data(start_date='2023-01-01', end_date='2025-05-31')

if len(df) == 0:
    print("ERROR: No training data extracted")
    exit(1)

print(f"   Extracted {len(df)} training examples")

# Prepare features
print("\n2. Preparing features (including new BookmakerProb)...")
X, y, df_with_features = model.prepare_features(df)

print(f"   Feature matrix shape: {X.shape}")

# Split into train/test (80/20 split)
print("\n3. Splitting data into train/test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split temp into validation (for calibration) and test
# Use 50/50 split of the 20%: 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples (for calibration): {len(X_val)}")
print(f"   Test samples: {len(X_test)}")

# Train model
print("\n4. Training XGBoost model...")
model.train_model(X_train, y_train, X_test, y_test)

# Calibrate predictions on validation set
print("\n5. Calibrating predictions...")
model.calibrate_predictions(X_val, y_val)

# Save model
print("\n6. Saving model...")
model.save_model()

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nModel metrics:")
for metric, value in model.model_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nFeatures used:")
for i, feature in enumerate(model.feature_columns, 1):
    print(f"  {i}. {feature}")

print("\n✅ Model saved and ready for backtesting!")
print("Next: Run 'python backtest_staking_strategies.py' to test the improved model")
