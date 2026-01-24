"""
BACKTEST: Price2Min Strategy
- Period: 2024-2025
- Thresholds: BACK >= 0.60 (Variable), LAY >= 0.70 (Variable)
- EXECUTION POINT: 2 Minutes Out (Price2Min)
- EXECUTION OFFSET: None (LTP)
- BACK Strategy: Odds $2-$40, 4% Target Profit
- LAY Strategy: Odds < $30, Max Liability 10% of Bank
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
    print("BACKTEST: PRICE 2 MIN STRATEGY")
    print("Period: 2024-2025")
    print("EXECUTION POINT: 2 Minutes before race")
    print("Settings:")
    print("  - BACK: Thresholds (Sliding), Odds $2.00 - $40.00")
    print("  - LAY: Thresholds (Sliding HVol), Odds < $30.00")
    print("="*80)
    
    # 1. LOAD DATA - Select Price2Min instead of Price5Min
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        ge.Price2Min,
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
        (df['Price2Min'] > 0)
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
    
    # CALCULATE DISCREPANCY USING PRICE 2 MIN
    df_bt['Discrepancy'] = df_bt['Price2Min'] / df_bt['V41_Price']
    df_bt['Price_Diff'] = df_bt['Price2Min'] - df_bt['V41_Price']
    
    # Rename for model input consistency (Model expects Price5Min but we feed Price2Min as 'Current Price')
    # Actually, the model features were trained on 'Price5Min', but for backtesting 'Price2Min' strategy, 
    # we should check if we re-feed 'Price5Min' or 'Price2Min'. 
    # Logic: The Live Bot normally runs continuously. If we are making decisions at 2 mins, we feed Price2Min.
    # So we rename Price2Min -> Price5Min just for feature mapping, OR just construct the feature vector correctly.
    # The models V42/V43 use 'Price5Min' as a feature name.
    
    df_bt['Price5Min'] = df_bt['Price2Min'] # Hack for model feature name compatibility
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    # 3b. SIGNALS (Using Price2Min via 'Price5Min' variable or direct)
    df_bt['Signal'] = 'PASS'
    
    # BACK Logic: Sliding Scale (User Defined)
    def get_back_signal(row):
        p = row['Price2Min']
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
    def get_lay_signal(row):
        p = row['Price2Min']
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
    
    lay_mask = (df_bt['IsLay']) & (~df_bt['RaceID'].isin(excluded_lay_races)) & (df_bt['Signal'] != 'BACK')
    df_bt.loc[lay_mask, 'Signal'] = 'LAY'
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    # 4. SIMULATION
    bets = []
    curr_bank = 200.0
    start_bank = 200.0
    
    for idx, row in df_bt.iterrows():
        signal = row['Signal']
        if signal == 'PASS': continue
            
        stake = 0.0
        risk = 0.0
        
        curr_price = row['Price2Min']
        if not curr_price or curr_price <= 1.01: continue
        
        # EXECUTION: LTP (No Offset)
        exec_price = curr_price
        
        if signal == 'BACK':
            # 4% Target Profit
            target = curr_bank * 0.04
            if exec_price > 1.01:
                stake = target / (exec_price - 1.0)
            risk = stake
            
        elif signal == 'LAY':
            # 10% Liability Cap
            risk = curr_bank * 0.10
            if exec_price > 1.01:
                stake = risk / (exec_price - 1.0)
            else:
                stake = 0.0 
                
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
            'Price': curr_price,
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
    print("RESULTS: PRICE 2 MIN STRATEGY")
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
