
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

DB_PATH = 'greyhound_racing.db'
MODEL_PATH = 'src/models/lay_model.pkl'

def train_lay_model():
    print("Loading Data for Lay Model...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Get Tier 1 Tracks
    try:
        with open('tier1_tracks.txt', 'r') as f:
            safe_tracks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("tier1_tracks.txt not found, using all tracks")
        safe_tracks = []

    # 2. Extract Data
    if safe_tracks:
        placeholders = ',' .join('?' for _ in safe_tracks)
        track_filter = f"AND t.TrackName IN ({placeholders})"
        params = safe_tracks
    else:
        track_filter = ""
        params = []

    query = f"""
    SELECT
        ge.GreyhoundID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Position
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      {track_filter}
    """
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    print(f"Loaded {len(df)} rows.")

    # 3. Preprocessing (Exact match to optimize_lay_strategy.py)
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    # Calculate Benchmarks
    print("Calculating Track/Dist Benchmarks...")
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    
    # Merge Benchmarks
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    # Calculate DogNormTimeAvg
    print("Calculating Feature: DogNormTimeAvg...")
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Drop NAs
    train_df = df.dropna(subset=['DogNormTimeAvg', 'NormTime'])
    print(f"Total dataset size: {len(train_df)}")

    # Split Train/Test (Walk Forward)
    # Train on data before 2025-01-01, Test on 2025+
    cutoff_date = pd.Timestamp('2025-01-01')
    train_set = train_df[train_df['MeetingDate'] < cutoff_date]
    test_set = train_df[train_df['MeetingDate'] >= cutoff_date]
    
    print(f"Training Data (pre-2025): {len(train_set)} rows")
    print(f"Test Data (2025+): {len(test_set)} rows")
    
    # 4. Train Model
    print("Training XGBRegressor...")
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror', 
        n_estimators=100, 
        n_jobs=-1, 
        tree_method='hist'
    )
    model.fit(train_set[features], train_set['NormTime'])
    
    # 5. Evaluate
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    preds_train = model.predict(train_set[features])
    preds_test = model.predict(test_set[features])
    
    metrics = {
        'train_mae': mean_absolute_error(train_set['NormTime'], preds_train),
        'test_mae': mean_absolute_error(test_set['NormTime'], preds_test),
        'test_rmse': np.sqrt(mean_squared_error(test_set['NormTime'], preds_test)),
        'test_r2': r2_score(test_set['NormTime'], preds_test),
        'train_samples': len(train_set),
        'test_samples': len(test_set)
    }
    
    print("\nModel Metrics:")
    print(f"  Test MAE:  {metrics['test_mae']:.4f}s")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}s")
    print(f"  Test R2:   {metrics['test_r2']:.4f}")

    # 5b. Retrain on ALL Data for Production
    print("\nTraining Final Production Model (Full Dataset)...")
    final_model = xgb.XGBRegressor(
        objective='reg:absoluteerror', 
        n_estimators=100, 
        n_jobs=-1, 
        tree_method='hist'
    )
    # Train on existing train_df which has NAs dropped
    final_model.fit(train_df[features], train_df['NormTime'])
    
    # 6. Save Artifacts (Model + Benchmarks + Metrics)
    artifacts = {
        'model': final_model, # Save the production model
        'benchmarks': bench.set_index(['TrackName', 'Distance'])['MedianTime'].to_dict(),
        'metrics': metrics,
        'feature_importance': dict(zip(features, final_model.feature_importances_))
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)
        
    print(f"Model and Benchmarks saved to {MODEL_PATH}")
    return metrics

if __name__ == "__main__":
    train_lay_model()
