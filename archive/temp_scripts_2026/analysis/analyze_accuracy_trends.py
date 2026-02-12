"""
ANALYSIS: Prediction Accuracy by Threshold (Under $40 Cap)
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def analyze_accuracy():
    print("="*60)
    print("ANALYSIS: PREDICTION ACCURACY (ODDS <= $40)")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, ge.LTP,
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.Price5Min IS NOT NULL AND ge.Price5Min > 0
    AND ge.LTP IS NOT NULL AND ge.LTP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Filter: Price Cap $40
    df = df[df['Price5Min'] <= 40.0].copy()
    print(f"Population (2024-2025, Odds <= $40, Valid LTP): {len(df)}")
    
    model_v41 = joblib.load(MODEL_V41)
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)
    
    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    def safe_predict(model, data_df, cols=None):
        if cols: data_df = data_df[cols]
        
        if hasattr(model, 'predict_proba'):
            # Sklearn Wrapper
            return model.predict_proba(data_df)[:, 1]
        else:
            # Raw Booster
            dmat = xgb.DMatrix(data_df)
            return model.predict(dmat)

    df['V41_Prob'] = safe_predict(model_v41, df, v41_cols)
    df['V41_Price'] = 1.0 / df['V41_Prob']
    
    df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
    df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    df['Steam_Prob'] = safe_predict(model_v42, df, alpha_feats)
    df['Drift_Prob'] = safe_predict(model_v43, df, alpha_feats)
    
    # Ground Truth (Using Actual LTP)
    print("Using Actual LTP for Accuracy Check...")
    # Steamer: Price5Min > LTP (Price dropped)
    df['Actual_Steam'] = (df['Price5Min'] > df['LTP']).astype(int)
    # Drifter: Price5Min < LTP (Price rose)
    df['Actual_Drift'] = (df['Price5Min'] < df['LTP']).astype(int)
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    print("\n--- V42 STEAMER ACCURACY ---")
    print(f"{'Threshold':<10} | {'Signals':<8} | {'Correct':<8} | {'Accuracy %':<10}")
    print("-" * 45)
    for th in thresholds:
        subset = df[df['Steam_Prob'] >= th]
        count = len(subset)
        if count > 0:
            correct = subset['Actual_Steam'].sum()
            acc = (correct / count) * 100
            print(f"{th:<10.2f} | {count:<8} | {correct:<8} | {acc:<10.2f}")
        else:
            print(f"{th:<10.2f} | {0:<8} | {0:<8} | {'N/A':<10}")

    print("\n--- V43 DRIFTER ACCURACY ---")
    print(f"{'Threshold':<10} | {'Signals':<8} | {'Correct':<8} | {'Accuracy %':<10}")
    print("-" * 45)
    for th in thresholds:
        subset = df[df['Drift_Prob'] >= th]
        count = len(subset)
        if count > 0:
            correct = subset['Actual_Drift'].sum()
            acc = (correct / count) * 100
            print(f"{th:<10.2f} | {count:<8} | {correct:<8} | {acc:<10.2f}")
        else:
            print(f"{th:<10.2f} | {0:<8} | {0:<8} | {'N/A':<10}")

if __name__ == "__main__":
    analyze_accuracy()
