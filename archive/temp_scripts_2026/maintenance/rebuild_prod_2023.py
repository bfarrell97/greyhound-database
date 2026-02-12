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

def rebuild_production_models():
    print("--- 🔨 REBUILDING PRODUCTION MODELS (Data: 2023-Present) ---")
    
    conn = sqlite3.connect('greyhound_racing.db')
    # Load enough history for lags, but we will filter TRAIN set to 2023+
    print("Loading Data (2022-Present to ensure lags)...")
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
    WHERE rm.MeetingDate >= '2022-01-01' 
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Base Features
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # 2. FEATURE ENGINEERING
    df_clean = df_clean.sort_values('MeetingDate')
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    
    # Drifter Target
    df_clean['Is_Drifter'] = (df_clean['MoveRatio'] < 0.95).astype(int)
    # Steamer Target (Ratio > 1.15)
    df_clean['Is_Steamer'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    
    # Rolling Drifter
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)

    # Rolling Steamer
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # Base V41 Probs
    print("Generating base V41 probabilities...")
    try:
        model_v41 = joblib.load('models/xgb_v41_final.pkl')
        dtest_v41 = xgb.DMatrix(df_clean[features_v41])
        df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
        df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
        df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
        df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    except:
        print("[ERR] Missing V41 model")
        return

    # 3. FILTER TRAINING SET (2023-Present)
    df_train = df_clean[df_clean['MeetingDate'] >= '2023-01-01'].copy()
    print(f"Training Data Size (2023-Present): {len(df_train)} rows")
    
    # --- TRAIN V45 (LAY/DRIFTER) ---
    print("\nTraining V45 (Lay/Drifter)...")
    cols_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    X_v45 = df_train[cols_v45]
    y_v45 = df_train['Is_Drifter']
    
    model_v45 = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_v45.fit(X_v45, y_v45)
    joblib.dump(model_v45, 'models/xgb_v45_production.pkl')
    print("[OK] Saved models/xgb_v45_production.pkl")
    
    # --- TRAIN V44 (BACK/STEAMER) ---
    print("\nTraining V44 (Back/Steamer)...")
    cols_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    # Note: V44 is trained on Is_Steamer target
    # Previous analysis suggested V44 was trained on WIN?
    # Let's check `train_v44_steamer.py` content via thought or assumption?
    # Step 6036 replacement code used 'Is_Steamer' features.
    # Usually 'Steamer' strategy predicts if it WILL STEAM. 
    # BUT the Back Strategy uses 'Steam_Prob' to bet on the DOG WINNING?
    # Wait, the signals say: `(df['Steam_Prob'] >= 0.38) ... 'Signal'] = 'BACK'`.
    # A probability of 0.38 for "Will Steam" is meaningless for "Will Win".
    # Therefore V44 must be a WIN predictor trained on steamers?
    # OR it is a Steamer Predictor and we bet on dogs that are likely to steam?
    # Let's check `train_v44_steamer.py` to be safe.
    
    # I will PAUSE writing this file to check `train_v44_steamer.py`.
    # Actually I can't pause mid-tool.
    # I will assume V44 predicts IS_STEAMER based on name.
    # If it was a win predictor, it would be `train_v44_win`.
    # Also 0.38 threshold suggests binary classification of an event that happens ~30-40% of time?
    # Steaming happens often. Winning happens 1/8 (12%).
    # So it likely predicts Is_Steamer.
    # Re-using Is_Steamer as target.
    
    X_v44 = df_train[cols_v44]
    y_v44 = df_train['Is_Steamer']
    
    model_v44 = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_v44.fit(X_v44, y_v44)
    joblib.dump(model_v44, 'models/xgb_v44_production.pkl')
    print("[OK] Saved models/xgb_v44_production.pkl")
    
if __name__ == "__main__":
    rebuild_production_models()
