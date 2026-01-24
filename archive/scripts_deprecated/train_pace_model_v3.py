"""
Train Pace Model V3 - Enhanced
New features: Track-specific form, Consistency, Box performance, Recency weighting
Hyperparameter optimization via early stopping
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import pickle
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def train_pace_v3():
    conn = sqlite3.connect(DB_PATH)
    
    progress("Loading data...")
    query = """
    SELECT
        ge.GreyhoundID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Position,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTime IS NOT NULL 
      AND ge.FinishTime > 15 AND ge.FinishTime < 50
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND rm.MeetingDate >= '2019-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    progress(f"Loaded {len(df):,} records")
    
    # BENCHMARKS
    progress("Calculating benchmarks...")
    benchmarks = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
    df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedian']
    
    # SORT
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    progress("Calculating features...")
    
    # --- STANDARD FEATURES ---
    df['Lag1'] = g['NormTime'].shift(1)
    df['Lag2'] = g['NormTime'].shift(2)
    df['Lag3'] = g['NormTime'].shift(3)
    
    df['Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['Roll10'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    
    # --- NEW FEATURES ---
    
    # 1. CONSISTENCY: Std of last 5 runs (lower = more consistent)
    df['Std5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std()).fillna(0.5)
    
    # 2. FORM TREND: Is dog improving or declining?
    df['Trend'] = df['Roll3'] - df['Roll10']  # Negative = recent form is better than long-term
    
    # 3. RECENCY WEIGHTED AVERAGE: More weight on recent runs
    def exp_weighted_avg(x, span=5):
        return x.shift(1).ewm(span=span, min_periods=3).mean()
    df['ExpRoll5'] = g['NormTime'].transform(lambda x: exp_weighted_avg(x, span=5))
    
    # 4. TRACK-SPECIFIC FORM: How does this dog perform at THIS track?
    # Create a key for dog+track
    df['DogTrack'] = df['GreyhoundID'].astype(str) + '_' + df['TrackName']
    df = df.sort_values(['DogTrack', 'MeetingDate'])
    gt = df.groupby('DogTrack')
    df['TrackRoll3'] = gt['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    df['TrackRoll3'] = df['TrackRoll3'].fillna(df['Roll5'])  # Fall back to overall form
    
    # 5. DISTANCE-SPECIFIC FORM: Sprinter vs Stayer
    df['DistCat'] = pd.cut(df['Distance'], bins=[0, 400, 550, 1000], labels=['Sprint', 'Middle', 'Stay'])
    df['DogDist'] = df['GreyhoundID'].astype(str) + '_' + df['DistCat'].astype(str)
    df = df.sort_values(['DogDist', 'MeetingDate'])
    gd = df.groupby('DogDist')
    df['DistRoll3'] = gd['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    df['DistRoll3'] = df['DistRoll3'].fillna(df['Roll5'])
    
    # 6. CLASS INDICATOR: Career Prize money (log scale)
    df['CareerPrize'] = df.groupby('GreyhoundID')['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df['LogPrize'] = np.log1p(df['CareerPrize'])
    
    # 7. WIN RATE (calculate before re-sorting messes up index)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['WinRate'] = df.groupby('GreyhoundID')['IsWin'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.125)
    
    # 8. BEST TIME: Fastest ever run (before this race)
    df['BestTime'] = df.groupby('GreyhoundID')['NormTime'].transform(lambda x: x.shift(1).expanding().min())
    df['BestTime'] = df['BestTime'].fillna(df['Roll5'])
    
    # Re-sort by original order
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # DROP NAs
    model_df = df.dropna(subset=['Roll5', 'ExpRoll5']).copy()
    progress(f"Training set size: {len(model_df):,} records")
    
    # FEATURES LIST
    features = [
        'Lag1', 'Lag2', 'Lag3',          # Raw lags
        'Roll3', 'Roll5', 'Roll10',       # Rolling averages
        'ExpRoll5',                        # Exponential weighted
        'Std5', 'Trend',                   # Consistency & trend
        'TrackRoll3', 'DistRoll3',         # Track/Distance specific
        'LogPrize', 'WinRate', 'BestTime', # Class indicators
        'DaysSince', 'Box', 'Distance'     # Race context
    ]
    target = 'NormTime'
    
    # TRAIN/TEST SPLIT
    train_mask = model_df['MeetingDate'] < '2024-01-01'
    test_mask = model_df['MeetingDate'] >= '2024-01-01'
    
    X_train = model_df.loc[train_mask, features]
    y_train = model_df.loc[train_mask, target]
    X_test = model_df.loc[test_mask, features]
    y_test = model_df.loc[test_mask, target]
    
    progress(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # TRAIN WITH EARLY STOPPING
    progress("Training XGBoost V3...")
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # EVALUATE
    progress("Evaluating...")
    preds = model.predict(X_test)
    mae_model = mean_absolute_error(y_test, preds)
    r2_model = r2_score(y_test, preds)
    
    # Baseline: Simple roll5
    mae_baseline = mean_absolute_error(y_test, X_test['Roll5'])
    
    print("\n" + "="*60)
    print("PACE MODEL V3 RESULTS (2024-2025 Test Set)")
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
    
    # Save
    with open('models/pace_v3_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nSaved to models/pace_v3_xgb_model.pkl")
    
    return mae_model, mae_baseline

if __name__ == "__main__":
    train_pace_v3()
