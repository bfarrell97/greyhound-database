"""
Retrain Production Models on FULL DATA (2020-Present)
- V44 Steamer (Back)
- V45 Drifter (Lay)
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def retrain_all():
    print("--- RETRAINING PRODUCTION MODELS (FULL DATA: 2020-Present) ---")
    
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, 
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2020-01-01' 
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows (2020-Present)")
    
    # 1. Base Features (V41)
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    df_clean = df_clean.sort_values('MeetingDate')
    
    # 2. Targets
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    df_clean['Is_Drifter'] = (df_clean['MoveRatio'] < 0.95).astype(int)
    
    # 3. Rolling Features (Steamer)
    print("Engineering Steam Lags...")
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 4. Rolling Features (Drifter)
    print("Engineering Drift Lags...")
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 5. Base V41 Probabilities
    print("Generating V41 Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 6. Feature Lists
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    features_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    # Use ALL DATA for training
    print(f"Training Set: {len(df_clean)} rows (FULL DATA)")
    
    # --- TRAIN V44 (STEAMER / BACK) ---
    print("\nTraining V44 Production (Steamer)...")
    X_v44 = df_clean[features_v44]
    y_v44 = df_clean['Is_Steamer']
    
    model_v44 = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_v44.fit(X_v44, y_v44)
    joblib.dump(model_v44, 'models/xgb_v44_production.pkl')
    print("[OK] Saved models/xgb_v44_production.pkl")
    
    # --- TRAIN V45 (DRIFTER / LAY) ---
    print("\nTraining V45 Production (Drifter)...")
    X_v45 = df_clean[features_v45]
    y_v45 = df_clean['Is_Drifter']
    
    model_v45 = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_v45.fit(X_v45, y_v45)
    joblib.dump(model_v45, 'models/xgb_v45_production.pkl')
    print("[OK] Saved models/xgb_v45_production.pkl")
    
    print("\n--- DONE ---")

if __name__ == "__main__":
    retrain_all()
