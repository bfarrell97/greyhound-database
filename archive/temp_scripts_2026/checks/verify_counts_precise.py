import sqlite3
import pandas as pd
import joblib
import xgboost as xgb
import sys
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"

def check():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        ge.Price5Min,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.GreyhoundName as Dog, g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01' AND rm.MeetingDate <= '2025-12-31'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Filter for valid price
    df = df[df['Price5Min'] > 0].copy()
    
    model_v41 = joblib.load(MODEL_V41)
    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    dmatrix = xgb.DMatrix(df[v41_cols])
    df['V41_Prob'] = model_v41.predict(dmatrix)
    df['V41_Price'] = 1.0 / df['V41_Prob']
    
    df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
    df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    model_v42 = joblib.load(MODEL_V42)
    df['Steam_Prob'] = model_v42.predict_proba(df[alpha_feats])[:, 1]
    
    # Check Counts
    total_signals = len(df[df['Steam_Prob'] >= 0.60])
    signals_under_40 = len(df[(df['Steam_Prob'] >= 0.60) & (df['Price5Min'] <= 40.0)])
    
    print(f"Total Signals >= 0.60: {total_signals}")
    print(f"Signals >= 0.60 AND Price <= $40: {signals_under_40}")
    print(f"Signals >= 0.60 AND Price > $40: {total_signals - signals_under_40}")

if __name__ == "__main__":
    check()
