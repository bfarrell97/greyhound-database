"""
TEST: Sub-$2 Odds Check
"""
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

def check_sub_2():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.Price5Min, ge.BSP, rm.MeetingDate,
        ge.Split, ge.FinishTime, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, g.GreyhoundName as Dog, g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01' AND ge.Price5Min > 0 AND ge.Price5Min < 2.0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
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
    
    steamers = df[df['Steam_Prob'] >= 0.60]
    
    print(f"\nSub-$2 Analysis (Threshold 0.60):")
    print(f"Bets Found: {len(steamers)}")
    if not steamers.empty:
        steamers['Win'] = (pd.to_numeric(steamers['Position'], errors='coerce') == 1).astype(int)
        
        # Simulate simple fixed staking
        stake = 100
        pnl = 0
        for _, row in steamers.iterrows():
            if row['Win'] == 1:
                pnl += stake * (row['Price5Min'] - 1) * 0.95
            else:
                pnl -= stake
        
        roi = pnl / (len(steamers) * stake) * 100
        print(f"PnL: ${pnl:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Win Rate: {steamers['Win'].mean()*100:.1f}%")

if __name__ == "__main__":
    check_sub_2()
