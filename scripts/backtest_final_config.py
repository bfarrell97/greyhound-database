"""
BACKTEST: Final Custom Configuration
- Period: 2024-2025
- Thresholds: BACK >= 0.60, LAY >= 0.70 (NO TRAP)
- BACK Strategy: Odds $2-$30, 4% Target Profit
- LAY Strategy: Odds < $30, Max Liability 10% of Bank
- Compounding Bank
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

def run_backtest():
    print("="*80)
    print("BACKTEST: FINAL CUSTOM CONFIGURATION")
    print("Period: 2024-2025")
    print("Settings:")
    print("  - BACK: Threshold >= 0.60 (No Trap), Odds $2.00 - $40.00")
    print("  - BACK Staking: 4% Target Profit (Compounding)")
    print("  - LAY: Threshold >= 0.70, Odds < $40.00")
    print("  - LAY Staking: Max Liability 10% of Bank (Compounding)")
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
    
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['MeetingDate'] <= '2025-12-31') & 
        (df['Price5Min'] > 0)
    ].sort_values('MeetingDate').copy()
    
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
    
    df_bt['Signal'] = 'PASS'
    # BACK Logic: >= 0.60 (No Trap)
    df_bt.loc[df_bt['Steam_Prob'] >= 0.60, 'Signal'] = 'BACK'
    # LAY Logic: >= 0.65
    df_bt.loc[df_bt['Drift_Prob'] >= 0.65, 'Signal'] = 'LAY'
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    # 4. SIMULATION
    bets = []
    curr_bank = 200.0
    start_bank = 200.0
    
    # EXCLUSION RULE: Max 2 Lays per Race
    # Identify races where we would have triggered > 2 Lays
    lay_mask = (df_bt['Drift_Prob'] >= 0.65) & (df_bt['Price5Min'] < 30.0)
    lay_candidates = df_bt[lay_mask]
    race_lay_counts = lay_candidates['RaceID'].value_counts()
    
    print("\n--- Lay Bets per Race Distribution ---")
    dist = race_lay_counts.value_counts().sort_index()
    for count, frequency in dist.items():
        print(f"   {count} Lays: {frequency} races")
    print("--------------------------------------")
    
    excluded_lay_races = set(race_lay_counts[race_lay_counts > 2].index)
    print(f"\n[Rule] Excluding {len(excluded_lay_races)} races with > 2 Lay candidates.")

    print("\nSimulating Bets...")
    
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal == 'PASS': continue
        
        curr_price = row['Price5Min']
        # COMPLEX LOGIC
        
        # COMPLEX LOGIC
        
        # BACK STRATEGY
        # Rule 1: Threshold >= 0.70 AND Odds <= $30 (High confidence, capped risk)
        # Rule 2: Threshold >= 0.60 AND Odds $5-40
        is_back = False
        if row['Steam_Prob'] >= 0.70:
            if curr_price <= 30.0: # Added Cap
                is_back = True
        elif row['Steam_Prob'] >= 0.60:
            if 5.0 <= curr_price <= 40.0:
                is_back = True
                
        if is_back:
            signal = 'BACK'
            
        # LAY STRATEGY
        # Rule: Threshold >= 0.65 AND Odds < $30
        is_lay = False
        if row['Drift_Prob'] >= 0.65:
            if curr_price < 30.0:
                if row['RaceID'] not in excluded_lay_races:
                    is_lay = True
                
        if is_lay:
            signal = 'LAY'
            
        # Overwrite signal from dataframe with new logic
        if is_back: signal = 'BACK'
        elif is_lay: signal = 'LAY'
        else: signal = 'PASS'
        
        if signal == 'PASS': continue
            
        stake = 0.0
        risk = 0.0
        
        # DETERMINE EXECUTION PRICE
        # default to Price5Min
        exec_price = curr_price 
        
        if signal == 'LAY':
            # User Request: Use Price5Min (Default)
            exec_price = curr_price
            if not exec_price or exec_price <= 1.0:
                continue
        
        # STAKING (Compounding)
        if signal == 'BACK':
            # 4% Target Profit
            target = curr_bank * 0.04
            if exec_price > 1.01:
                stake = target / (exec_price - 1.0)
            risk = stake
        else:
            # 10% Liability Cap
            risk = curr_bank * 0.10
            if exec_price > 1.01:
                stake = risk / (exec_price - 1.0)
                
        # SAFETY: Don't bet more than we have
        if risk > curr_bank:
            risk = curr_bank
            if signal == 'BACK': stake = risk
            else: stake = risk / (exec_price - 1.0)

        # OUTCOME
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                profit = stake * (exec_price - 1) * 0.95 # 5% Comm
                pnl = profit
            else:
                pnl = -stake
        else: # LAY
            if win:
                pnl = -risk
            else:
                pnl = stake * 0.95 # 5% Comm on stake won
        
        curr_bank += pnl
        
        bets.append({
            'Date': row['MeetingDate'],
            'Signal': signal,
            'Price': curr_price,
            'Steam_Prob': row['Steam_Prob'],
            'Drift_Prob': row['Drift_Prob'],
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl,
            'Bank': curr_bank,
            'Win': 1 if pnl > 0 else 0
        })
        
        # Prevent bankruptcy
        if curr_bank <= 5.0:
            print("❌ BANKRUPTCY!")
            break

    # 5. REPORT
    res = pd.DataFrame(bets)
    
    if res.empty:
        print("\nNo bets generated.")
        return
    
    print("\n" + "="*60)
    print("RESULTS: FINAL CONFIGURATION")
    print("="*60)
    print(f"Final Bank: ${curr_bank:.2f} (Start: ${start_bank:.2f})")
    print(f"Total Profit: ${curr_bank - start_bank:.2f}")
    
    print(f"\nTotal Bets: {len(res)}")
    
    for sig in ['BACK', 'LAY']:
        sub = res[res['Signal'] == sig]
        if sub.empty: continue
        stake = sub['Stake'].sum()
        risk = sub['Risk'].sum()
        pnl = sub['PnL'].sum()
        wins = sub['Win'].sum()
        roi = (pnl / risk * 100) if risk > 0 else 0
        print(f"  {sig}: {len(sub)} bets | Win Rate: {wins/len(sub)*100:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
        
    print(f"\n{'='*40}")
    
    res['Year'] = res['Date'].dt.year
    yearly = res.groupby('Year').agg({
        'PnL': 'sum', 
        'Risk': 'sum', 
        'Signal': 'count'
    }).rename(columns={'Signal': 'Bets'})
    yearly['ROI'] = yearly['PnL'] / yearly['Risk'] * 100
    print("\n--- By Year ---")
    print(yearly.to_string())
    
    res.to_csv("backtest_final_config.csv", index=False)
    print("\nSaved to backtest_final_config.csv")

if __name__ == "__main__":
    run_backtest()
