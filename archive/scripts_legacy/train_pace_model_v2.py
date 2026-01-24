"""
Train Pace Model V2 (Advanced Features)
Adds Split History, Weight, Consistency, and Strike Rate to the XGBoost Model.
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

DB_PATH = 'greyhound_racing.db'

def train_pace_v2():
    print("Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query with extra columns
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split,
        ge.FinishTime,
        ge.Weight,
        ge.Position,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FinishTime IS NOT NULL
      AND ge.Split IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND rm.MeetingDate >= '2020-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(30.0) # Default avg weight
    
    # 1. Benchmarks (Global)
    print("Calculating Benchmarks...")
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    # Targets
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    # 2. Advanced Feature Engineering
    print("Feature Engineering...")
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # --- Existing Features ---
    # Pace Lags/Rolls
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Std5']  = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).std().fillna(0)) # Consistency
    
    # --- New Features ---
    # 1. Split History (Does Split predict Pace?)
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    
    # 2. Weight (Trend)
    # Is dog racing heavier or lighter than usual?
    df['AvgWeight'] = g['Weight'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['WeightDiff'] = df['Weight'] - df['AvgWeight'].fillna(df['Weight'])
    
    # 3. Win Rate (Strike Rate)
    # 1 if Position='1', 0 else
    df['IsWin'] = (df['Position'] == '1').astype(int)
    # Global career win rate up to this race
    df['WinRate'] = g['IsWin'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    # 4. Days Since
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    
    # Filter valid
    model_df = df.dropna(subset=['p_Roll5', 's_Roll3']).copy()
    print(f"Dataset Size: {len(model_df)}")
    
    # 3. Train/Test Split
    train_mask = model_df['MeetingDate'] < '2024-01-01'
    test_mask = model_df['MeetingDate'] >= '2024-01-01'
    
    features = [
        'p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'p_Std5', # Pace History
        's_Lag1', 's_Roll3',                                          # Split History
        'Weight', 'WeightDiff',                                       # Weight
        'WinRate',                                                    # Class
        'DaysSince', 'Box', 'Distance'
    ]
    target = 'NormTime'
    
    X_train = model_df.loc[train_mask, features]
    y_train = model_df.loc[train_mask, target]
    
    X_test = model_df.loc[test_mask, features]
    y_test = model_df.loc[test_mask, target]
    
    print(f"Training on {len(X_train)} rows with {len(features)} features...")
    
    # 4. Train XGBoost
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*60)
    print("PACE MODEL V2 RESULTS")
    print("="*60)
    print(f"MAE: {mae:.4f} seconds (Baseline V1 was ~0.2654)")
    print(f"R2:  {r2:.4f}")
    
    print("\nFeature Importance:")
    imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    print(imp.to_string(index=False))
    
    # Save
    with open('models/pace_v2_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nSaved to models/pace_v2_xgb_model.pkl")

if __name__ == "__main__":
    train_pace_v2()
