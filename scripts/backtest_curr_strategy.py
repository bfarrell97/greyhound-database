
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import warnings

# Add root so we can import src
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    print("Could not import FeatureEngineerV41")
    sys.exit(1)

warnings.filterwarnings('ignore')

# CONFIG
DB_PATH = "C:/Users/Winxy/Documents/grold/greyhound-database/greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42_STE = "models/xgb_v42_steamer.pkl"
MODEL_V43_DRI = "models/xgb_v43_drifter.pkl"

def run_backtest():
    print("="*80)
    print("HIGH-FIDELITY V41 + ALPHA BACKTEST")
    print("Exact replication of Live Strategy")
    print("="*80)
    
    # 1. LOAD DATA
    print("[1/6] Loading Data & History...")
    conn = sqlite3.connect(DB_PATH)
    
    # Needs wide history for feature engineering
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
    """ # Load extra history for rolling, backtest starts 2024
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data.")
        return

    # 2. FEATURE ENGINEERING (V41 Exact)
    print("[2/6] Engineering V41 Features (This takes a moment)...")
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Filter to Backtest Window (2024+)
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['Price5Min'] > 0) & 
        (df['Price15Min'] > 0)
    ].copy()
    
    print(f"Backtest Population: {len(df_bt)} runs")

    # 3. V41 PROBABILITY GENERATION
    print("[3/6] Running V41 Super Model...")
    model_v41 = joblib.load(MODEL_V41)
    v41_cols = fe.get_feature_list()
    
    # Handle missing
    for c in v41_cols:
        if c not in df_bt.columns: df_bt[c] = 0
        df_bt[c] = pd.to_numeric(df_bt[c], errors='coerce').fillna(0)
        
    dmatrix_v41 = xgb.DMatrix(df_bt[v41_cols])
    df_bt['V41_Prob'] = model_v41.predict(dmatrix_v41)
    
    # 4. ALPHA SIGNALS GENERATION
    print("[4/6] Running Alpha Models (V42/V43)...")
    
    # CORRECT LOGIC (Matches predict_market_v42_v43.py)
    df_bt['V41_Price'] = 1.0 / df_bt['V41_Prob']
    df_bt['Discrepancy'] = df_bt['Price5Min'] / df_bt['V41_Price'] # Ratio
    df_bt['Price_Diff'] = df_bt['Price5Min'] - df_bt['V41_Price'] # Difference
    
    # Alpha Features
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    model_v42 = joblib.load(MODEL_V42_STE)
    model_v43 = joblib.load(MODEL_V43_DRI)
    
    df_bt['V42_Score'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['V43_Score'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    # 5. EXECUTION SIMULATION
    print("[5/6] Simulating Staking...")
    
    bets = []
    
    # FIX: Position Check
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    for idx, row in df_bt.iterrows():
        signal = None
        
        # Exact Logic from PROD SCRIPT (predict_market_v42_v43.py)
        # Back: >= 0.60 (But < 0.70 Trap)
        # Lay: >= 0.70
        
        if row['V42_Score'] >= 0.53:
            if row['V42_Score'] > 0.70:
                pass # STEAMER TRAP
            else:
                signal = 'BACK'
                
        elif row['V43_Score'] >= 0.55:
            signal = 'LAY'  
        
        if not signal: continue
            
        curr_price = row['Price5Min']
        
        # ODDS CAPS
        if signal == 'BACK' and curr_price > 15.0: continue
        if signal == 'LAY' and curr_price > 15.0: continue 

        # STAKING (Strict)
        stake = 0.0
        risk = 0.0
        
        if signal == 'BACK':
            # 4% Target Profit ($8 on $200)
            target = 8.0 
            if curr_price > 1.01:
                stake = target / (curr_price - 1.0)
            risk = stake
        else:
            # LAY: 6% Fixed Liability ($12 on $200)
            risk = 12.0
            if curr_price > 1.01:
                stake = risk / (curr_price - 1.0)
            
            # Note: We simulate placing ALL micro-stakes
            
        # OUTCOME
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                # Win: (Stake * Price) - Stake - Comm
                # Settle at Price5Min (Limit taken)
                profit = stake * (curr_price - 1) * 0.95 # 5% Comm
                pnl = profit
            else:
                pnl = -stake
        else:
            if win:
                # Layer Loses: Pay Liability
                pnl = -risk 
            else:
                # Layer Wins: Keep Stake - Comm
                profit = stake * 0.95
                pnl = profit
                
        bets.append({
            'Signal': signal,
            'Price': curr_price,
            'Stake': stake,
            'PnL': pnl,
            'V41Prob': row['V41_Prob']
        })

    # 6. REPORT
    res = pd.DataFrame(bets)
    if res.empty:
        print("No bets generated.")
        return
        
    print("\n" + "="*40)
    print("RESULTS (High Fidelity V41)")
    print("="*40)
    print(res.groupby('Signal')[['Stake', 'PnL']].sum())
    
    print(f"\nTotal PnL: ${res['PnL'].sum():.2f}")
    
    # Save
    res.to_csv("backtest_fidelity_results.csv")
    print("\nSaved to backtest_fidelity_results.csv")

if __name__ == "__main__":
    run_backtest()
