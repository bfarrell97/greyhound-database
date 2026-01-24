import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys

# Ensure src import
sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def load_data():
    print("Loading Data (2020-2025)...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Selecting all needed columns for V41
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2020-01-01'
    ORDER BY rm.MeetingDate ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_v41():
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    # Feature Engineering
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    # Prepare Data
    # Be strict on NaN to ensure quality
    df_clean = df.dropna(subset=features)
    print(f"Training set (Clean): {len(df_clean)} rows")
    
    # Time Series Split (Train until Oct 2024, Test Oct24-Jan25)
    df_clean['MeetingDate'] = pd.to_datetime(df_clean['MeetingDate'])
    test_start = pd.Timestamp('2024-10-01') 
    
    train = df_clean[df_clean['MeetingDate'] < test_start]
    test = df_clean[df_clean['MeetingDate'] >= test_start]
    
    # Ensure Test only contains valid BSP for evaluation fairness? 
    # Actually just train on all, eval on all.
    
    X_train = train[features]
    y_train = train['win']
    
    X_test = test[features]
    y_test = test['win']
    bsp_test = test['BSP']
    
    print(f"Train Sz: {len(X_train)}, Test Sz: {len(X_test)}")
    
    # Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.03, # Lower learning rate better for complex interactions
        'max_depth': 7,        # Slightly deeper for interactions
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("Training Model V41...")
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1500, 
        evals=[(dtest, 'Test')], 
        early_stopping_rounds=50, 
        verbose_eval=100
    )
    
    # Save Feature Importance
    gain = model.get_score(importance_type='gain')
    print("\nFeature Importance (Gain):")
    for k, v in sorted(gain.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.2f}")

    # Evaluate against BSP (Quick Check)
    if 'BSP' in df.columns:
        preds = model.predict(dtest)
        mask = bsp_test > 1.0
        if mask.sum() > 0:
            from sklearn.metrics import log_loss, roc_auc_score
            y_valid = y_test[mask]
            p_valid = preds[mask]
            bsp_valid = 1.0 / bsp_test[mask]
            
            auc_v41 = roc_auc_score(y_valid, p_valid)
            auc_bsp = roc_auc_score(y_valid, bsp_valid)
            
            print("\n-------------------------------------------")
            print("V41 vs MARKET (Oct 24 - Jan 25)")
            print(f"V41 AUC: {auc_v41:.4f}")
            print(f"BSP AUC: {auc_bsp:.4f}")
            print("-------------------------------------------")

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_v41_final.pkl')
    print("Model saved to models/xgb_v41_final.pkl")

if __name__ == "__main__":
    train_v41()
