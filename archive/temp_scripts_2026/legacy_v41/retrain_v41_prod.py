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

def retrain_v41_prod():
    print("RUNNING V41 PRODUCTION RETRAINING (FULL DATASET)...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load all available data (2020-2025)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate < date('now')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features + ['win']).copy()
    
    X = df_clean[features]
    y = df_clean['win']
    
    print(f"Retraining on {len(df_clean)} samples (Full Historical Data)...")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    
    out_path = 'models/xgb_v41_final_prod.pkl'
    joblib.dump(model, out_path)
    print(f"SUCCESS: Production model saved to {out_path}")

if __name__ == "__main__":
    retrain_v41_prod()
    input("\nPress Enter to exit...")
