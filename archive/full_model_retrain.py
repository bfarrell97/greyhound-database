"""
Full model retraining with updated 9-feature system including LastN_AvgFinishBenchmark
This trains the complete model on historical data and validates performance
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Import the updated model class
from greyhound_ml_model import GreyhoundMLModel

def get_database_connection():
    """Get connection to SQLite database"""
    db_path = 'greyhound_racing.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    return sqlite3.connect(db_path)

def load_training_data():
    """Load all available training data from database"""
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    conn = get_database_connection()
    query = """
    SELECT 
        r.RaceID, rm.MeetingDate as RaceDate, t.TrackName as Track, r.Distance, ge.Box,
        ge.StartingPrice as Odds, 
        (CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as Win,
        ge.GreyhoundID, ge.Weight,
        ge.SplitBenchmarkLengths,
        ge.FinishTimeBenchmarkLengths
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate IS NOT NULL
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
    ORDER BY rm.MeetingDate
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"✓ Loaded {len(df):,} total race entries")
    print(f"  Date range: {df['RaceDate'].min()} to {df['RaceDate'].max()}")
    print(f"  Unique races: {df['RaceID'].nunique():,}")
    print(f"  Unique greyhounds: {df['GreyhoundID'].nunique():,}")
    
    # Convert Position to numeric
    df['Win'] = pd.to_numeric(df['Win'], errors='coerce').fillna(0)
    print(f"  Win rate: {df['Win'].mean()*100:.2f}%")
    
    return df

def prepare_training_data(df):
    """Prepare data for model training with new features"""
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    # Sort by date
    df = df.sort_values('RaceDate').reset_index(drop=True)
    df['RaceDate'] = pd.to_datetime(df['RaceDate'])
    
    training_data = []
    
    # Process each greyhound's history
    greyhound_ids = df['GreyhoundID'].unique()
    print(f"Processing {len(greyhound_ids):,} greyhounds...")
    
    for idx, greyhound_id in enumerate(greyhound_ids):
        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1:,}/{len(greyhound_ids):,} greyhounds processed")
        
        greyhound_df = df[df['GreyhoundID'] == greyhound_id].copy()
        
        # Calculate rolling features for each race
        for i in range(len(greyhound_df)):
            if i < 5:  # Need at least 5 races of history
                continue
            
            race = greyhound_df.iloc[i]
            history = greyhound_df.iloc[max(0, i-5):i]  # Last 5 races
            
            # Calculate historical features
            win_rate = history['Win'].mean() if len(history) > 0 else 0
            
            # CRITICAL: Calculate LastN_AvgFinishBenchmark
            finish_benchmarks = history['FinishTimeBenchmarkLengths'].dropna()
            if len(finish_benchmarks) > 0:
                avg_finish_benchmark = finish_benchmarks.mean()
            else:
                avg_finish_benchmark = 0
            
            training_data.append({
                'RaceID': race['RaceID'],
                'GreyhoundID': greyhound_id,
                'RaceDate': race['RaceDate'],
                'Track': race['Track'],
                'Distance': race['Distance'],
                'Box': race['Box'],
                'Weight': race['Weight'],
                'Odds': race['Odds'],
                'Win': race['Win'],
                # Features
                'LastN_WinRate': win_rate,
                'LastN_AvgFinishBenchmark': avg_finish_benchmark,  # NEW FEATURE
            })
    
    training_df = pd.DataFrame(training_data)
    print(f"\n✓ Created {len(training_df):,} training examples")
    print(f"  Win rate: {training_df['Win'].mean()*100:.2f}%")
    print(f"  Date range: {training_df['RaceDate'].min()} to {training_df['RaceDate'].max()}")
    
    return training_df

def split_data(df, test_start_date='2025-09-01'):
    """Split into train and test sets"""
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    test_start = pd.to_datetime(test_start_date)
    
    train_df = df[df['RaceDate'] < test_start].copy()
    test_df = df[df['RaceDate'] >= test_start].copy()
    
    print(f"Training set: {len(train_df):,} examples")
    print(f"  Date range: {train_df['RaceDate'].min()} to {train_df['RaceDate'].max()}")
    print(f"  Win rate: {train_df['Win'].mean()*100:.2f}%")
    
    print(f"\nTest set: {len(test_df):,} examples")
    print(f"  Date range: {test_df['RaceDate'].min()} to {test_df['RaceDate'].max()}")
    print(f"  Win rate: {test_df['Win'].mean()*100:.2f}%")
    
    return train_df, test_df

def train_model(train_df, test_df):
    """Train XGBoost model with new features"""
    print("\n" + "="*80)
    print("TRAINING MODEL (with 9 features)")
    print("="*80)
    
    # Feature columns
    feature_cols = ['Box', 'Distance', 'Weight', 'LastN_WinRate', 'LastN_AvgFinishBenchmark']
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Win']
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Win']
    
    print(f"Features: {feature_cols}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    print("✓ Model training complete")
    
    # Feature importance
    print("\nFeature Importance:")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp*100:.2f}%")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Confidence analysis
    high_conf = y_pred_proba >= 0.5
    if high_conf.sum() > 0:
        high_conf_accuracy = (y_pred[high_conf] == y_test[high_conf]).mean()
        print(f"High Confidence (>=0.5): {high_conf.sum():,} predictions, {high_conf_accuracy*100:.2f}% accuracy")
    
    return model, feature_cols

def validate_on_odds_ranges(test_df, model, feature_cols):
    """Validate model performance on different odds ranges"""
    print("\n" + "="*80)
    print("VALIDATION ON ODDS RANGES")
    print("="*80)
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['Win']
    
    # Convert Odds to numeric
    test_df_copy = test_df.copy()
    test_df_copy['Odds'] = pd.to_numeric(test_df_copy['Odds'], errors='coerce')
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Test on different odds ranges
    odds_ranges = [
        (1.50, 2.00, "$1.50-$2.00"),
        (1.50, 2.50, "$1.50-$2.50"),
        (2.00, 3.00, "$2.00-$3.00"),
        (1.50, 3.00, "$1.50-$3.00"),
    ]
    
    for min_odds, max_odds, label in odds_ranges:
        mask = (test_df_copy['Odds'] >= min_odds) & (test_df_copy['Odds'] <= max_odds)
        
        if mask.sum() == 0:
            continue
        
        range_y_test = y_test[mask]
        range_pred_proba = y_pred_proba[mask]
        range_odds = test_df_copy[mask]['Odds']
        
        # High confidence bets (>=0.5)
        high_conf = range_pred_proba >= 0.5
        
        if high_conf.sum() > 0:
            wins = range_y_test[high_conf].sum()
            total = high_conf.sum()
            strike_rate = wins / total * 100
            
            # Calculate ROI
            stakes = total * 1.0  # Stake $1 per bet
            returns = wins * range_odds[high_conf].mean()
            roi = (returns - stakes) / stakes * 100
            
            print(f"\n{label}:")
            print(f"  Predictions: {total:,} bets (strike: {strike_rate:.1f}%)")
            print(f"  ROI: {roi:+.2f}%")
            print(f"  Expected return: ${returns:,.0f} on ${stakes:,.0f}")

def save_model(model, feature_cols):
    """Save model for deployment"""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    model_path = 'greyhound_ml_model_retrained.pkl'
    feature_path = 'model_features_retrained.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Features saved to {feature_path}")

def main():
    print("\n" + "="*80)
    print("FULL MODEL RETRAINING WITH 9-FEATURE SYSTEM")
    print("="*80)
    
    # Load and prepare data
    df = load_training_data()
    training_df = prepare_training_data(df)
    train_df, test_df = split_data(training_df)
    
    # Train model
    model, feature_cols = train_model(train_df, test_df)
    
    # Validate
    validate_on_odds_ranges(test_df, model, feature_cols)
    
    # Save
    save_model(model, feature_cols)
    
    print("\n" + "="*80)
    print("RETRAINING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Compare new model performance vs baseline 8-feature model")
    print("2. Deploy greyhound_ml_model_retrained.pkl for live predictions")
    print("3. Monitor real-world ROI improvement")

if __name__ == "__main__":
    main()
