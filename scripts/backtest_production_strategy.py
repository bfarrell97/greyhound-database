"""
BACKTEST: Current Production Strategy (V42/V43)
Exact replication of predict_market_v42_v43.py thresholds.
Period: 2024-2025
NO MODIFICATIONS TO PRODUCTION CODE.
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

# PRODUCTION THRESHOLDS (FROM predict_market_v42_v43.py - DO NOT CHANGE)
BACK_THRESHOLD = 0.60
BACK_TRAP_THRESHOLD = 0.70  # Steamer Trap - don't bet if above this
LAY_THRESHOLD = 0.70
ODDS_CAP = 15.0

def run_backtest(start_date='2024-01-01', end_date='2025-12-31'):
    print("="*80)
    print("BACKTEST: PRODUCTION V42/V43 STRATEGY")
    print(f"Period: {start_date} to {end_date}")
    print(f"Thresholds: BACK >= {BACK_THRESHOLD} (Trap > {BACK_TRAP_THRESHOLD}), LAY >= {LAY_THRESHOLD}")
    print("="*80)
    
    # 1. LOAD DATA
    print("[1/5] Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    query = f"""
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
        (df['MeetingDate'] >= start_date) & 
        (df['MeetingDate'] <= end_date) & 
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
    
    # 4. ALPHA SIGNALS (V42/V43)
    print("[4/5] Running Alpha Models...")
    
    # Features (EXACT MATCH to production)
    df_bt['Discrepancy'] = df_bt['Price5Min'] / df_bt['V41_Price']
    df_bt['Price_Diff'] = df_bt['Price5Min'] - df_bt['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    # Signal Logic (EXACT MATCH to production)
    df_bt['Signal'] = 'PASS'
    df_bt.loc[df_bt['Steam_Prob'] >= BACK_THRESHOLD, 'Signal'] = 'BACK'
    df_bt.loc[df_bt['Steam_Prob'] > BACK_TRAP_THRESHOLD, 'Signal'] = 'STEAMER TRAP (PASS)'
    df_bt.loc[df_bt['Drift_Prob'] >= LAY_THRESHOLD, 'Signal'] = 'LAY'
    
    # 5. SIMULATE BETS
    print("[5/5] Simulating Staking...")
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    bets = []
    BANKROLL = 200.0
    
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal in ['PASS', 'STEAMER TRAP (PASS)']: continue
        
        curr_price = row['Price5Min']
        
        # Odds Cap
        if curr_price > ODDS_CAP: continue
        
        # Staking
        stake = 0.0
        risk = 0.0
        
        if signal == 'BACK':
            target = BANKROLL * 0.04  # 4% Target Profit
            if curr_price > 1.01:
                stake = target / (curr_price - 1.0)
            risk = stake
        else:  # LAY
            risk = BANKROLL * 0.06  # 6% Fixed Liability
            if curr_price > 1.01:
                stake = risk / (curr_price - 1.0)
        
        # Outcome
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                pnl = stake * (curr_price - 1) * 0.95  # 5% Comm
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
            'Steam_Prob': row['Steam_Prob'],
            'Drift_Prob': row['Drift_Prob'],
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl,
            'Win': 1 if pnl > 0 else 0
        })

    # REPORT
    res = pd.DataFrame(bets)
    
    if res.empty:
        print("\nNo bets generated with current thresholds.")
        return
    
    print("\n" + "="*60)
    print("RESULTS: PRODUCTION V42/V43 STRATEGY")
    print("="*60)
    
    print(f"\nTotal Bets: {len(res)}")
    
    # By Signal Type
    print("\n--- By Signal Type ---")
    for sig in ['BACK', 'LAY']:
        sub = res[res['Signal'] == sig]
        if sub.empty: continue
        stake = sub['Stake'].sum()
        risk = sub['Risk'].sum()
        pnl = sub['PnL'].sum()
        wins = sub['Win'].sum()
        roi = (pnl / risk * 100) if risk > 0 else 0
        print(f"  {sig}: {len(sub)} bets | Win Rate: {wins/len(sub)*100:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
    
    # Overall
    total_pnl = res['PnL'].sum()
    total_risk = res['Risk'].sum()
    total_roi = (total_pnl / total_risk * 100) if total_risk > 0 else 0
    total_wins = res['Win'].sum()
    
    print(f"\n{'='*40}")
    print(f"TOTAL: {len(res)} bets | Win Rate: {total_wins/len(res)*100:.1f}%")
    print(f"Total PnL: ${total_pnl:.2f} | ROI: {total_roi:.2f}%")
    print(f"{'='*40}")
    
    # By Year
    print("\n--- By Year ---")
    res['Year'] = res['Date'].dt.year
    yearly = res.groupby('Year').agg({'PnL': 'sum', 'Risk': 'sum', 'Signal': 'count'}).rename(columns={'Signal': 'Bets'})
    yearly['ROI'] = yearly['PnL'] / yearly['Risk'] * 100
    print(yearly.to_string())
    
    # Save
    output_file = f"backtest_production_{start_date[:4]}_{end_date[:4]}.csv"
    res.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2025-12-31'
    run_backtest(start, end)
