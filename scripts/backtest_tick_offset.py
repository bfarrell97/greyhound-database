"""
BACKTEST: Tick Offset Strategy
- Period: 2024-2025
- Thresholds: BACK >= 0.60 (Variable), LAY >= 0.70 (Variable)
- BACK Strategy: Odds $2-$40, 4% Target Profit
- LAY Strategy: Odds < $30, Max Liability 10% of Bank
- EXECUTION OFFSET:
  - LAY: Price5Min + 2 Ticks (Conservative)
  - BACK: Price5Min - 1 Tick (Conservative)
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41
from src.integration.betfair_fetcher import BetfairOddsFetcher
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def run_backtest():
    print("="*80)
    print("BACKTEST: TICK OFFSET STRATEGY")
    print("Period: 2024-2025")
    print("Settings:")
    print("  - BACK: Thresholds (Sliding), Odds $2.00 - $40.00")
    print("  - BACK EXECUTION: Price5Min - 1 Tick")
    print("  - LAY: Thresholds (Sliding HVol), Odds < $30.00")
    print("  - LAY EXECUTION: Price5Min + 2 Ticks")
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
    
    # 3b. SIGNALS
    df_bt['Signal'] = 'PASS'
    
    # BACK Logic: Sliding Scale (User Defined)
    # $1-2: 0.60, $2-6: 0.55, $6-10: 0.60, $10-40: 0.70
    def get_back_signal(row):
        p = row['Price5Min']
        prob = row['Steam_Prob']
        thresh = 0.99
        if p < 2.0: thresh = 0.60
        elif p < 6.0: thresh = 0.55
        elif p < 10.0: thresh = 0.60
        elif p <= 40.0: thresh = 0.70
        
        if prob >= thresh: return True
        return False
        
    df_bt['IsBack'] = df_bt.apply(get_back_signal, axis=1)
    
    # LAY Logic: High Volume Standard
    # <4: 0.55, 4-8: 0.60, >8: 0.65
    def get_lay_signal(row):
        p = row['Price5Min']
        prob = row['Drift_Prob']
        thresh = 0.65
        if p < 4.0: thresh = 0.55
        elif p < 8.0: thresh = 0.60
        else: thresh = 0.65
        
        if prob >= thresh and p < 30.0: return True
        return False

    df_bt['IsLay'] = df_bt.apply(get_lay_signal, axis=1)
    
    # EXCLUSION RULE: Max 2 Lays per Race
    race_lay_counts = df_bt[df_bt['IsLay']]['RaceID'].value_counts()
    excluded_lay_races = set(race_lay_counts[race_lay_counts > 2].index)
    
    df_bt.loc[df_bt['IsBack'], 'Signal'] = 'BACK'
    # Prioritize Back? Or keep distinct? Usually Back > Lay if conflict?
    # Logic: If Back signal exists, take Back. If Lay signal exists (and not excluded), take Lay.
    # What if both? 
    # Let's say Back overrides default.
    
    lay_mask = (df_bt['IsLay']) & (~df_bt['RaceID'].isin(excluded_lay_races)) & (df_bt['Signal'] != 'BACK')
    df_bt.loc[lay_mask, 'Signal'] = 'LAY'
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    # 4. SIMULATION
    bets = []
    curr_bank = 200.0
    start_bank = 200.0
    
    # Initialize Fetcher for Math (No Login)
    fetcher = BetfairOddsFetcher()
    
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal == 'PASS': continue
            
        stake = 0.0
        risk = 0.0
        
        curr_price = row['Price5Min']
        if not curr_price or curr_price <= 1.01: continue
        
        # DETERMINE EXECUTION PRICE WITH TICK OFFSET
        exec_price = curr_price
        
        if signal == 'BACK':
            # Offset: -1 Tick (Conservative)
            exec_price = fetcher.get_next_tick(curr_price, -1)
            
            # 4% Target Profit
            target = curr_bank * 0.04
            if exec_price > 1.01:
                stake = target / (exec_price - 1.0)
            risk = stake
            
        elif signal == 'LAY':
            # Offset: +2 Ticks (Conservative)
            exec_price = fetcher.get_next_tick(curr_price, 2)
            
            # 10% Liability Cap
            risk = curr_bank * 0.10
            if exec_price > 1.01:
                stake = risk / (exec_price - 1.0)
            else:
                stake = 0.0 # Safety
                
        # SAFETY CHECK
        if risk > curr_bank:
            risk = curr_bank
            if signal == 'BACK': stake = risk
            else: stake = risk / (exec_price - 1.0)
            
        # PnL CALCULATION
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
            'BasePrice': curr_price,
            'ExecPrice': exec_price,
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl,
            'Bank': curr_bank,
            'Win': 1 if pnl > 0 else 0
        })
        
        if curr_bank <= 5.0:
            print("❌ BANKRUPTCY!")
            break

    # 5. REPORT
    res = pd.DataFrame(bets)
    if res.empty:
        print("\nNo bets generated.")
        return
    
    print("\n" + "="*60)
    print("RESULTS: TICK OFFSET CONFIGURATION")
    print("="*60)
    print(f"Final Bank: ${curr_bank:.2f} (Start: ${start_bank:.2f})")
    print(f"Total Profit: ${curr_bank - start_bank:.2f}")
    
    for sig in ['BACK', 'LAY']:
        sub = res[res['Signal'] == sig]
        if sub.empty: continue
        risk = sub['Risk'].sum()
        pnl = sub['PnL'].sum()
        wins = sub['Win'].sum()
        roi = (pnl / risk * 100) if risk > 0 else 0
        print(f"  {sig}: {len(sub)} bets | Win Rate: {wins/len(sub)*100:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
        
    print(f"\n{'='*40}")
    
    res['Year'] = pd.to_datetime(res['Date']).dt.year
    yearly = res.groupby('Year').agg({
        'PnL': 'sum', 
        'Risk': 'sum', 
        'Signal': 'count'
    }).rename(columns={'Signal': 'Bets'})
    yearly['ROI'] = yearly['PnL'] / yearly['Risk'] * 100
    print("\n--- By Year ---")
    print(yearly.to_string())

if __name__ == "__main__":
    run_backtest()
