"""
BACKTEST: Raw Ratio Strategy (Original Holy Grail Analysis)
Bypasses V42/V43 models - uses Ratio > 1.05 directly.
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

def run_backtest():
    print("="*80)
    print("BACKTEST: RAW RATIO STRATEGY (Holy Grail Replication)")
    print("Using Ratio > 1.05 for BACK, Ratio < 0.95 for LAY")
    print("="*80)
    
    # 1. LOAD DATA
    print("[1/5] Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        ge.Price5Min, ge.Price15Min,
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
    
    if df.empty:
        print("No data.")
        return

    # 2. FEATURE ENGINEERING (V41)
    print("[2/5] Engineering V41 Features...")
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Filter to Backtest Window
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['Price5Min'] > 0)
    ].copy()
    
    print(f"Backtest Population: {len(df_bt)} runs")

    # 3. V41 PROBABILITY
    print("[3/5] Running V41 Super Model...")
    model_v41 = joblib.load(MODEL_V41)
    v41_cols = fe.get_feature_list()
    
    for c in v41_cols:
        if c not in df_bt.columns: df_bt[c] = 0
        df_bt[c] = pd.to_numeric(df_bt[c], errors='coerce').fillna(0)
        
    dmatrix = xgb.DMatrix(df_bt[v41_cols])
    df_bt['V41_Prob'] = model_v41.predict(dmatrix)
    df_bt['V41_Price'] = 1.0 / df_bt['V41_Prob']
    
    # 4. RAW RATIO CALCULATION
    print("[4/5] Calculating Ratio & Simulating Bets...")
    df_bt['Ratio'] = df_bt['Price5Min'] / df_bt['V41_Price']
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    bets = []
    BANKROLL = 200.0
    
    for idx, row in df_bt.iterrows():
        signal = None
        ratio = row['Ratio']
        curr_price = row['Price5Min']
        
        # RAW RATIO LOGIC (Holy Grail)
        if ratio > 1.05:
            signal = 'BACK'  # Steamer: Market overpriced vs V41
        elif ratio < 0.95:
            signal = 'LAY'   # Drifter: Market underpriced vs V41
            
        if not signal: continue
        
        # ODDS CAPS
        if curr_price > 15.0: continue
        
        # STAKING
        stake = 0.0
        risk = 0.0
        
        if signal == 'BACK':
            target = BANKROLL * 0.04  # 4% Target Profit
            if curr_price > 1.01:
                stake = target / (curr_price - 1.0)
            risk = stake
        else:
            risk = BANKROLL * 0.06  # 6% Fixed Liability
            if curr_price > 1.01:
                stake = risk / (curr_price - 1.0)
        
        # OUTCOME
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                profit = stake * (curr_price - 1) * 0.95  # 5% Comm
                pnl = profit
            else:
                pnl = -stake
        else:  # LAY
            if win:
                pnl = -risk
            else:
                pnl = stake * 0.95
                
        bets.append({
            'Date': row['MeetingDate'],
            'Signal': signal,
            'Price': curr_price,
            'Ratio': ratio,
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl
        })

    # 5. REPORT
    print("[5/5] Generating Report...")
    res = pd.DataFrame(bets)
    
    if res.empty:
        print("No bets generated.")
        return
        
    print("\n" + "="*60)
    print("RESULTS: RAW RATIO STRATEGY")
    print("="*60)
    
    print(f"\nTotal Bets: {len(res)}")
    print(f"  - BACK (Ratio > 1.05): {len(res[res['Signal']=='BACK'])}")
    print(f"  - LAY (Ratio < 0.95): {len(res[res['Signal']=='LAY'])}")
    
    print("\n--- By Signal Type ---")
    summary = res.groupby('Signal').agg({
        'Stake': 'sum',
        'Risk': 'sum',
        'PnL': 'sum'
    })
    print(summary)
    
    # Calculate ROI
    for sig in ['BACK', 'LAY']:
        sig_df = res[res['Signal']==sig]
        if not sig_df.empty:
            total_risk = sig_df['Risk'].sum()
            total_pnl = sig_df['PnL'].sum()
            roi = (total_pnl / total_risk * 100) if total_risk > 0 else 0
            print(f"\n{sig} ROI: {roi:.2f}%")
    
    total_pnl = res['PnL'].sum()
    total_risk = res['Risk'].sum()
    total_roi = (total_pnl / total_risk * 100) if total_risk > 0 else 0
    
    print(f"\n=== TOTAL PnL: ${total_pnl:.2f} | ROI: {total_roi:.2f}% ===")
    
    res.to_csv("backtest_raw_ratio_results.csv", index=False)
    print("\nSaved to backtest_raw_ratio_results.csv")

if __name__ == "__main__":
    run_backtest()
