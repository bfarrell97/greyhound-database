"""
Train PIR Prediction Model (XGBoost)
Objective: Predict 'NormSplit' (Split vs Track Median) accurately.
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import pickle

DB_PATH = 'greyhound_racing.db'

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def train_pir_model():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. LOAD DATA
    progress("Loading valid split data...")
    query = """
    SELECT
        ge.GreyhoundID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Split IS NOT NULL 
      AND ge.Split > 0 AND ge.Split < 30
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    
    progress(f"Loaded {len(df):,} records")
    
    # 2. FEATURE ENGINEERING
    progress("Calculating features...")
    
    # Benchmarks
    benchmarks = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
    df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
    
    # Target: NormSplit
    df['NormSplit'] = df['Split'] - df['TrackDistMedian']
    
    # Sort for lag features
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Lags
    df['Lag1'] = df.groupby('GreyhoundID')['NormSplit'].shift(1)
    df['Lag2'] = df.groupby('GreyhoundID')['NormSplit'].shift(2)
    df['Lag3'] = df.groupby('GreyhoundID')['NormSplit'].shift(3)
    
    # Rolling Avgs
    df['Roll3'] = df.groupby('GreyhoundID')['NormSplit'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=3).mean()
    )
    df['Roll5'] = df.groupby('GreyhoundID')['NormSplit'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Days Since Last
    df['PrevDate'] = df.groupby('GreyhoundID')['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days
    
    # Drop NaNs created by lags
    model_df = df.dropna(subset=['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince']).copy()
    
    progress(f"Training set size: {len(model_df):,} records")
    
    # 3. SPLIT TRAIN/TEST
    # Train: < 2024
    # Test: >= 2024
    train_mask = model_df['MeetingDate'] < '2024-01-01'
    test_mask = model_df['MeetingDate'] >= '2024-01-01'
    
    features = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    target = 'NormSplit'
    
    X_train = model_df.loc[train_mask, features]
    y_train = model_df.loc[train_mask, target]
    
    X_test = model_df.loc[test_mask, features]
    y_test = model_df.loc[test_mask, target]
    
    # 4. TRAIN XGBOOST
    progress("Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 5. EVALUATE
    progress("Evaluating performance...")
    
    # Model Predictions
    preds = model.predict(X_test)
    mae_model = mean_absolute_error(y_test, preds)
    r2_model = r2_score(y_test, preds)
    
    # Baseline Prediction (Simple Average of Lag1, 2, 3)
    # This simulates "Last 3 Avg" strategy
    baseline_preds = X_test[['Lag1', 'Lag2', 'Lag3']].mean(axis=1)
    mae_baseline = mean_absolute_error(y_test, baseline_preds)
    
    print("\n" + "="*60)
    print("PIR PREDICTION RESULTS (2024-2025 Test Set)")
    print("="*60)
    print(f"Model MAE:    {mae_model:.4f} seconds")
    print(f"Baseline MAE: {mae_baseline:.4f} seconds")
    
    improvement = (mae_baseline - mae_model) / mae_baseline * 100
    print(f"Improvement:  {improvement:.1f}%")
    print(f"Model R²:     {r2_model:.4f}")
    
    print("\n--- Feature Importance ---")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))
    
    # Save Model
    with open('models/pir_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to models/pir_xgb_model.pkl")

if __name__ == "__main__":
    train_pir_model()
