"""
Train Pace Model V1 (Back Strategy)
====================================
Features: Lag1, Lag2, Lag3, Roll3, Roll5, DaysSince, Box, Distance
Target: NormTime (FinishTime - MedianTime)
Training: 2020-2025 data
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os


def train_pace_model():
    """Train the Pace Model V1 on 2020-2025 data"""
    print("="*70)
    print("TRAINING PACE MODEL V1 (BACK STRATEGY)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # 1. Load benchmarks
    print("\n[1/5] Loading benchmarks...")
    bench_df = pd.read_sql_query(
        "SELECT TrackName, Distance, MedianTime FROM Benchmarks",
        conn
    )
    print(f"Loaded {len(bench_df)} benchmarks")
    
    # 2. Load race data 2020-2025
    print("\n[2/5] Loading race data 2020-2025...")
    query = """
    SELECT 
        ge.GreyhoundID, ge.RaceID, rm.MeetingDate, t.TrackName,
        r.Distance, ge.Box, ge.FinishTime, ge.Position
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
      AND ge.FinishTime IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} race entries")
    
    # 3. Calculate NormTime
    print("\n[3/5] Calculating NormTime and features...")
    df = df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    df = df.dropna(subset=['NormTime'])
    
    # Convert types
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    # Sort by dog and date
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Calculate rolling features per dog
    g = df.groupby('GreyhoundID')
    df['Lag1'] = g['NormTime'].shift(1)
    df['Lag2'] = g['NormTime'].shift(2)
    df['Lag3'] = g['NormTime'].shift(3)
    df['Roll3'] = g['NormTime'].shift(1).transform(lambda x: x.rolling(3, min_periods=3).mean())
    df['Roll5'] = g['NormTime'].shift(1).transform(lambda x: x.rolling(5, min_periods=5).mean())
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.clip(upper=60).fillna(30)
    
    # Drop rows without enough history (need Roll5)
    df = df.dropna(subset=['Roll5'])
    print(f"Samples with 5+ prior races: {len(df):,}")
    
    # 4. Train/Test Split
    print("\n[4/5] Training XGBoost model...")
    feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    X = df[feature_cols]
    y = df['NormTime']
    
    # Split: 2020-2024 train, 2025 test
    train_mask = df['MeetingDate'] < '2025-01-01'
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:absoluteerror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain MAE: {train_mae:.4f}s")
    print(f"Test MAE:  {test_mae:.4f}s")
    print(f"Test RMSE: {test_rmse:.4f}s")
    print(f"Test R2:   {test_r2:.4f}")
    
    # Feature importance
    feat_imp = dict(zip(feature_cols, model.feature_importances_))
    print("\nFeature Importance:")
    for feat, imp in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    # 5. Save model
    print("\n[5/5] Saving model...")
    model_path = 'models/pace_xgb_model.pkl'
    os.makedirs('models', exist_ok=True)
    
    artifacts = {
        'model': model,
        'feature_columns': feature_cols,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'feature_importance': feat_imp
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Saved to {model_path}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return artifacts['metrics']


if __name__ == "__main__":
    train_pace_model()
