"""
TEST: Threshold 0.60 + NO ODDS CAP
Removing odds cap to see full signal volume.
NO CHANGES TO PRODUCTION CODE.
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

# TEST THRESHOLDS
BACK_THRESHOLD = 0.60
BACK_TRAP_THRESHOLD = 0.70
LAY_THRESHOLD = 0.70
ODDS_CAP = 999.0  # NO CAP

def run_backtest():
    print("="*80)
    print("TEST: THRESHOLD 0.60 + NO ODDS CAP")
    print(f"BACK >= {BACK_THRESHOLD} (Trap > {BACK_TRAP_THRESHOLD}), LAY >= {LAY_THRESHOLD}")
    print("Odds Cap: NONE")
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
    ].copy()
    
    print(f"Population: {len(df_bt)} runs")

    model_v41 = joblib.load(MODEL_V41)
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
    
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    df_bt['Signal'] = 'PASS'
    df_bt.loc[df_bt['Steam_Prob'] >= BACK_THRESHOLD, 'Signal'] = 'BACK'
    df_bt.loc[df_bt['Steam_Prob'] > BACK_TRAP_THRESHOLD, 'Signal'] = 'STEAMER TRAP (PASS)'
    df_bt.loc[df_bt['Drift_Prob'] >= LAY_THRESHOLD, 'Signal'] = 'LAY'
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    bets = []
    BANKROLL = 200.0
    
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal in ['PASS', 'STEAMER TRAP (PASS)']: continue
        
        curr_price = row['Price5Min']
        # NO ODDS CAP
        
        stake = 0.0
        risk = 0.0
        
        if signal == 'BACK':
            target = BANKROLL * 0.04
            if curr_price > 1.01:
                stake = target / (curr_price - 1.0)
            risk = stake
        else:
            risk = BANKROLL * 0.06
            if curr_price > 1.01:
                stake = risk / (curr_price - 1.0)
        
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                pnl = stake * (curr_price - 1) * 0.95
            else:
                pnl = -stake
        else:
            if win:
                pnl = -risk
            else:
                pnl = stake * 0.95
                
        bets.append({
            'Date': row['MeetingDate'],
            'Signal': signal,
            'Price': curr_price,
            'Steam_Prob': row['Steam_Prob'],
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl,
            'Win': 1 if pnl > 0 else 0
        })

    res = pd.DataFrame(bets)
    
    if res.empty:
        print("\nNo bets generated.")
        return
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
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
    
    total_pnl = res['PnL'].sum()
    total_risk = res['Risk'].sum()
    total_roi = (total_pnl / total_risk * 100) if total_risk > 0 else 0
    
    print(f"\n{'='*40}")
    print(f"TOTAL PnL: ${total_pnl:.2f} | ROI: {total_roi:.2f}%")
    print(f"{'='*40}")
    
    res['Year'] = res['Date'].dt.year
    yearly = res.groupby('Year').agg({'PnL': 'sum', 'Risk': 'sum', 'Signal': 'count'}).rename(columns={'Signal': 'Bets'})
    yearly['ROI'] = yearly['PnL'] / yearly['Risk'] * 100
    print("\n--- By Year ---")
    print(yearly.to_string())
    
    # Show price distribution of bets
    print("\n--- Price Distribution of BACK Bets ---")
    back_bets = res[res['Signal'] == 'BACK']
    if not back_bets.empty:
        print(back_bets['Price'].describe())

if __name__ == "__main__":
    run_backtest()
