"""
TEST: Staking Variations on 'Simple $40 Cap' Strategy
Signal Config: BACK 0.60 ($2-$40), LAY 0.70 (<$40)
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
    print("TEST: STAKING VARIATIONS")
    print("Config: BACK >= 0.60 ($2-$40), LAY >= 0.70 (<$40)")
    print("="*80)
    
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

    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['MeetingDate'] <= '2025-12-31') & 
        (df['Price5Min'] > 0)
    ].sort_values('MeetingDate').copy()
    
    print(f"Population: {len(df_bt)} runs")

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
    
    # GENERATE SIGNALS ONCE
    df_bt['Signal'] = 'PASS'
    df_bt.loc[df_bt['Steam_Prob'] >= 0.60, 'Signal'] = 'BACK'
    df_bt.loc[df_bt['Drift_Prob'] >= 0.70, 'Signal'] = 'LAY'  # Priority to LAY if conflict? (Rare overlap)
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    df_bt['Win'] = (df_bt['Position'] == 1).astype(int)
    
    # FILTER SIGNALS
    signals_df = []
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal == 'PASS': continue
        
        curr_price = row['Price5Min']
        if signal == 'BACK':
            if not (2.0 <= curr_price <= 40.0): continue
        elif signal == 'LAY':
            if curr_price >= 40.0: continue
            
        signals_df.append(row)
        
    signals_df = pd.DataFrame(signals_df)
    print(f"Total Signals: {len(signals_df)}")
    
    # SIMULATE STAKING PLANS
    plans = [
        {'name': 'Conservative', 'back_target': 0.02, 'lay_liab': 0.05, 'flat': 0},
        {'name': 'Current',      'back_target': 0.04, 'lay_liab': 0.10, 'flat': 0},
        {'name': 'Aggressive',   'back_target': 0.06, 'lay_liab': 0.15, 'flat': 0},
        {'name': 'Flat $10',     'back_target': 0,    'lay_liab': 0,    'flat': 10.0}
    ]
    
    results = []
    
    for plan in plans:
        bank = 200.0
        max_bank = 200.0
        drawdown = 0.0
        
        for idx, row in signals_df.iterrows():
            signal = row['Signal']
            price = row['Price5Min']
            win = (row['Win'] == 1)
            
            stake = 0.0
            risk = 0.0
            
            if plan['flat'] > 0:
                # Flat Staking
                if signal == 'BACK':
                    stake = plan['flat']
                    risk = stake
                else:
                    # Flat Stake for Lay? Or Flat Liability? assume Flat Liability for comparable safety
                    risk = plan['flat'] 
                    if price > 1.01: stake = risk / (price - 1.0)
            else:
                # Percentage Staking
                if signal == 'BACK':
                    target = bank * plan['back_target']
                    if price > 1.01: stake = target / (price - 1.0)
                    risk = stake
                else:
                    risk = bank * plan['lay_liab']
                    if price > 1.01: stake = risk / (price - 1.0)

            # Safety
            if risk > bank: 
                risk = bank
                if signal == 'BACK': stake = risk
                else: stake = risk / (price - 1.0)
                
            # Outcome
            pnl = 0.0
            if signal == 'BACK':
                if win: pnl = stake * (price - 1) * 0.95
                else: pnl = -stake
            else:
                if win: pnl = -risk
                else: pnl = stake * 0.95
            
            bank += pnl
            if bank > max_bank: max_bank = bank
            dd = (max_bank - bank) / max_bank * 100
            if dd > drawdown: drawdown = dd
            
            if bank <= 5: break
            
        results.append({
            'Plan': plan['name'],
            'End Bank': bank,
            'Profit': bank - 200,
            'Growth': (bank - 200) / 200 * 100,
            'Drawdown': drawdown
        })

    print("\nRESULTS COMPARISON:")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_backtest()
