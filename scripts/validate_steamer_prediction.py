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

def validate_steamer_prediction():
    print("Loading Data (2025) for Steamer Validation...")
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
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.Price5Min IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    
    # Predict V41
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df_clean[features])
    df_clean['Prob'] = model.predict(dtest)
    df_clean['ImpliedProb'] = 1.0 / df_clean['BSP']
    df_clean['Edge'] = df_clean['Prob'] - df_clean['ImpliedProb']
    
    # Define Target: Did it Steam? (BSP < Price5Min)
    df_clean['IsSteamer'] = (df_clean['BSP'] < df_clean['Price5Min']).astype(int)
    
    # V42 Heuristic: Negative Edge + Recognized Quality (Prob > 0.20)
    # i.e. "Market likes it (Price < 5.0) but Model thinks Price should be higher (Negative Edge)"
    df_clean['PredSteamer'] = ((df_clean['Edge'] < -0.10) & (df_clean['Prob'] > 0.20)).astype(int)
    
    preds = df_clean[df_clean['PredSteamer']==1]
    
    accuracy = preds['IsSteamer'].mean() * 100
    avg_price_drop = (preds['Price5Min'] - preds['BSP']).mean()
    
    print("\n" + "="*80)
    print("V42 STEAMER PREDICTION (Heuristic: Edge < -0.10 & Prob > 0.20)")
    print("="*80)
    print(f"Total Predicted Steamers: {len(preds)}")
    print(f"Accuracy (Actually Steamed): {accuracy:.2f}%")
    print(f"Avg Price Drop: ${avg_price_drop:.2f}")
    
    # Profitability Check (If we backed at 5Min Price)
    preds['Win'] = (preds['win'] == 1).astype(int)
    stake = 10
    preds['Ret_5Min'] = np.where(preds['Win']==1, stake*(preds['Price5Min']-1)*0.92+stake, 0)
    profit = preds['Ret_5Min'].sum() - (len(preds)*stake)
    roi = (profit / (len(preds)*stake)) * 100
    
    print("-" * 60)
    print("PROFITABILITY (Backing Predicted Steamers @ 5Min Price)")
    print(f"Profit: ${profit:.2f}")
    print(f"ROI:    {roi:+.2f}%")
    print("-" * 60)
    
    # Contrast with Random Baseline
    baseline_acc = df_clean['IsSteamer'].mean() * 100
    print(f"Baseline Steamer Probability (Random Dog): {baseline_acc:.2f}%")
    print(f"Lift: {accuracy / baseline_acc:.2f}x")

if __name__ == "__main__":
    validate_steamer_prediction()
