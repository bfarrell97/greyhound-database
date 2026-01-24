
import sqlite3
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def analyze():
    print("="*80)
    print("DRIFTER MODEL (V43): LOW ODDS ANALYSIS")
    print("="*80)
    
    # 1. LOAD DATA
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
    WHERE rm.MeetingDate >= '2023-06-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 2. FEATURES
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Filter for Analysis Period & Valid Prices
    df = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['Price5Min'] > 0)
    ].copy()
    
    # 3. PREDICT
    model_v43 = joblib.load(MODEL_V43)
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    # We need V41_Prob for the model features
    # (Assuming we can quickly load V41 or mock it if needed, but best to load real)
    try:
        model_v41 = joblib.load("models/xgb_v41_final.pkl")
        v41_cols = fe.get_feature_list()
        for c in v41_cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        import xgboost as xgb
        dmatrix = xgb.DMatrix(df[v41_cols])
        df['V41_Prob'] = model_v41.predict(dmatrix)
        df['V41_Price'] = 1.0 / df['V41_Prob']
        df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
        df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
    except:
        print("Error loading V41, using dummy V41 features (Not Ideal)")
        df['V41_Prob'] = 0.2
        df['Discrepancy'] = 1.0
        df['Price_Diff'] = 0.0

    df['Drift_Prob'] = model_v43.predict_proba(df[alpha_feats])[:, 1]
    
    # 4. ANALYSIS BREAKDOWN
    # Define Bands
    bands = [
        (1.0, 1.5, "Very Short ($1.00-$1.50)"),
        (1.5, 2.0, "Short ($1.50-$2.00)"),
        (2.0, 3.0, "Fav ($2.00-$3.00)"),
        (3.0, 4.0, "Mid ($3.00-$4.00)"),
        (4.0, 6.0, "Mid-High ($4.00-$6.00)"),
        (6.0, 999, "Long ($6.00+)")
    ]
    
    thresholds = [0.50, 0.55, 0.60, 0.65]
    
    print(f"\nTotal Rows: {len(df)}")
    
    for min_p, max_p, label in bands:
        print(f"\n>>> BAND: {label}")
        print(f"{'Threshold':<10} | {'Bets':<6} | {'Lay Win%':<10} | {'Drifted%':<10} | {'Avg PnL':<10} | {'ROI%':<8}")
        print("-" * 75)
        
        subset = df[(df['Price5Min'] >= min_p) & (df['Price5Min'] < max_p)]
        
        for thresh in thresholds:
            clips = subset[subset['Drift_Prob'] >= thresh].copy()
            if clips.empty:
                print(f"{thresh:<10} | {0:<6} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<8}")
                continue
                
            # Lay Outcome: Win if Position != 1
            clips['Lay_Win'] = clips['Position'] != 1
            
            # Drift Outcome: BSP > Price5Min
            clips['Did_Drift'] = clips['BSP'] > clips['Price5Min']
            
            # PnL (Flat 1u Stake)
            # If Win (Dog Lost): +0.95 (commission)
            # If Loss (Dog Won): -(Price - 1)
            clips['PnL'] = clips.apply(lambda x: 0.95 if x['Lay_Win'] else -(x['Price5Min'] - 1), axis=1)
            
            bets = len(clips)
            lay_win_rate = clips['Lay_Win'].mean() * 100
            drift_rate = clips['Did_Drift'].mean() * 100
            total_pnl = clips['PnL'].sum()
            liability = clips.apply(lambda x: (x['Price5Min'] - 1), axis=1).sum()
            # For ROI, usually PnL / Liability
            roi = (total_pnl / liability * 100) if liability > 0 else 0
            
            print(f"{thresh:<10} | {bets:<6} | {lay_win_rate:<9.1f}% | {drift_rate:<9.1f}% | ${total_pnl/bets:<8.2f} | {roi:<7.1f}%")

if __name__ == "__main__":
    analyze()
