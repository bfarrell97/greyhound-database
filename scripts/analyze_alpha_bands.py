"""
ANALYSIS: Alpha Thresholds & Odds Bands
Grid search for V42/V43 thresholds and reporting performance by Price Bands.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def run_analysis():
    print("="*80)
    print("ANALYSIS: ALPHA THRESHOLDS & ODDS BANDS")
    print("Period: 2024-2025")
    print("="*80)
    
    # 1. LOAD DATA
    print("Loading Data...")
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
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['MeetingDate'] <= '2025-12-31') & 
        (df['Price5Min'] > 0)
    ].copy()
    
    print(f"Population: {len(df_bt)} runs")

    # 3. MODELS
    model_v41 = joblib.load(MODEL_V41)
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)

    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df_bt.columns: df_bt[c] = 0
        df_bt[c] = pd.to_numeric(df_bt[c], errors='coerce').fillna(0)
        
    dmatrix = xgb.DMatrix(df_bt[v41_cols])
    df_bt['V41_Prob'] = model_v41.predict(dmatrix)
    df_bt['V41_Price'] = 1.0 / df_bt['V41_Prob']
    
    df_bt['Discrepancy'] = df_bt['Price5Min'] / df_bt['V41_Price']
    df_bt['Price_Diff'] = df_bt['Price5Min'] - df_bt['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    df_bt['Win'] = (pd.to_numeric(df_bt['Position'], errors='coerce') == 1).astype(int)
    
    # 4. ANALYSIS LOOP
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    bands = [
        (0, 5, "$0-$5"), 
        (5, 10, "$5-$10"), 
        (10, 20, "$10-$20"), 
        (20, 30, "$20-$30"),
        (30, 40, "$30-$40"),
        (40, 9999, "$40+")
    ]
    
    results = []
    
    print("\nRunning Grid Search...")
    
    for thres in thresholds:
        # BACK STRATEGY
        steamers = df_bt[df_bt['Steam_Prob'] >= thres].copy()
        for min_p, max_p, label in bands:
            band_df = steamers[(steamers['Price5Min'] >= min_p) & (steamers['Price5Min'] < max_p)]
            if len(band_df) == 0: continue
            
            # 4% Target Profit Staking
            stake_sum = 0
            pnl_sum = 0
            
            for _, row in band_df.iterrows():
                target = 200 * 0.04 # Fixed $8 target for standardization
                stake = target / (row['Price5Min'] - 1.0) if row['Price5Min'] > 1 else 0
                if row['Win'] == 1:
                    prof = stake * (row['Price5Min'] - 1) * 0.95
                    pnl_sum += prof
                else:
                    pnl_sum -= stake
                stake_sum += stake
                
            roi = (pnl_sum / stake_sum * 100) if stake_sum > 0 else 0
            results.append({
                'Strategy': 'BACK',
                'Threshold': thres,
                'Band': label,
                'Bets': len(band_df),
                'WinRate': band_df['Win'].mean() * 100,
                'PnL': pnl_sum,
                'ROI': roi
            })

        # LAY STRATEGY
        drifters = df_bt[df_bt['Drift_Prob'] >= thres].copy()
        for min_p, max_p, label in bands:
            band_df = drifters[(drifters['Price5Min'] >= min_p) & (drifters['Price5Min'] < max_p)]
            if len(band_df) == 0: continue
            
            # 10% Liability Staking
            stake_sum = 0
            risk_sum = 0
            pnl_sum = 0
            
            for _, row in band_df.iterrows():
                risk = 200 * 0.10 # Fixed $20 risk
                stake = risk / (row['Price5Min'] - 1.0) if row['Price5Min'] > 1 else 0
                
                if row['Win'] == 1:
                    pnl_sum -= risk
                else:
                    prof = stake * 0.95
                    pnl_sum += prof
                
                risk_sum += risk
                
            roi = (pnl_sum / risk_sum * 100) if risk_sum > 0 else 0
            results.append({
                'Strategy': 'LAY',
                'Threshold': thres,
                'Band': label,
                'Bets': len(band_df),
                'WinRate': band_df['Win'].mean() * 100,
                'PnL': pnl_sum,
                'ROI': roi
            })

    # 5. PRINT REPORT
    res_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("BACK STRATEGY ROI BY BAND")
    print("="*80)
    print(res_df[res_df['Strategy']=='BACK'].pivot(index='Threshold', columns='Band', values='ROI').round(1).to_string())
    print("\nBACK BET COUNTS")
    print(res_df[res_df['Strategy']=='BACK'].pivot(index='Threshold', columns='Band', values='Bets').fillna(0).astype(int).to_string())

    print("\n" + "="*80)
    print("LAY STRATEGY ROI BY BAND")
    print("="*80)
    print(res_df[res_df['Strategy']=='LAY'].pivot(index='Threshold', columns='Band', values='ROI').round(1).to_string())
    print("\nLAY BET COUNTS")
    print(res_df[res_df['Strategy']=='LAY'].pivot(index='Threshold', columns='Band', values='Bets').fillna(0).astype(int).to_string())
    
    res_df.to_csv("alpha_bands_analysis.csv", index=False)
    print("\nSaved detailed results to alpha_bands_analysis.csv")

if __name__ == "__main__":
    run_analysis()
