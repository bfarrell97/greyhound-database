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

def experiment_v45():
    print("--- 🧪 EXPERIMENT: TRAIN (2023-Jun 25) vs TEST (Jul-Nov 25) ---")
    
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
    
    # 1. Base Features (V41)
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # 2. Define DRIFTER Target (Ratio < 0.95: 5% Drift)
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Drifter'] = (df_clean['MoveRatio'] < 0.95).astype(int)
    
    # 3. Lag Features (Rolling Drift Context)
    print("Engineering Historical Drifter Features...")
    df_clean = df_clean.sort_values('MeetingDate')
    
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 4. Base Low-Level Probabilities (V41)
    print("Generating Base V41 Probabilities...")
    try:
        model_v41 = joblib.load('models/xgb_v41_final.pkl')
    except:
        print("[ERR] V41 Model missing. Cannot proceed.")
        return

    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 5. SPLIT DATA
    print("Splitting Data...")
    mask_train = (df_clean['MeetingDate'] >= '2023-01-01') & (df_clean['MeetingDate'] <= '2025-06-30')
    mask_test = (df_clean['MeetingDate'] >= '2025-07-01') & (df_clean['MeetingDate'] <= '2025-11-30')
    
    df_train = df_clean[mask_train].copy()
    df_test = df_clean[mask_test].copy()
    
    print(f"Train Set: {len(df_train)} rows (Jan 23 - Jun 25)")
    print(f"Test Set:  {len(df_test)} rows (Jul 25 - Nov 25)")
    
    features_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    X_train = df_train[features_v45]
    y_train = df_train['Is_Drifter']
    X_test = df_test[features_v45]
    y_test = df_test['Is_Drifter']
    
    print("Training XGBoost V45 (Exp)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save for user to use if good
    joblib.dump(model, 'models/xgb_v45_jun25.pkl')
    print("Model saved to models/xgb_v45_jun25.pkl")
    
    # 6. EVALUATE
    print("\n--- RESULTS ON TEST SET (Jul-Nov 2025) ---")
    probs = model.predict_proba(X_test)[:, 1]
    
    # Price Cap Mask
    price_cap_mask = df_test['Price5Min'] < 30.0
    
    print(f"{'Threshold':<10} | {'Volume':<8} | {'Daily':<6} | {'Precision':<8}")
    print("-" * 50)
    
    for t in [0.60, 0.63, 0.65, 0.70]:
        mask = (probs >= t) & price_cap_mask.values
        n = mask.sum()
        prec = y_test[mask].mean() if n > 0 else 0.0
        daily = n / (153) # Approx 5 months = 153 days
        print(f"{t:<10} | {n:<8} | {daily:<6.1f} | {prec:.1%}")

if __name__ == "__main__":
    experiment_v45()
