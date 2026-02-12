
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# CONFIG
DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_test_2020_2024.json"
COMMISSION = 0.10
BANKROLL = 200.0

# BACK CONFIG
BACK_MIN_PROB = 0.55
BACK_MIN_EDGE = 0.20
BACK_MAX_ODDS = 15.0
BACK_BOXES = [2, 3, 4, 5, 6, 7]

# LAY CONFIG
LAY_MAX_PROB = 0.20
LAY_MAX_EDGE = -0.48
LAY_MAX_PRICE = 2.00
LAY_BOXES = [1, 2]
STRAIGHT_TRACKS = ['Healesville', 'Murray Bridge (MBS)', 'Richmond (RIS)', 'Richmond Straight', 'Capalaba', 'Q Straight']

def run_hybrid_backtest():
    print("="*80)
    print("V33 HYBRID BACKTEST: SELECT on Price5Min -> SETTLE on BSP")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    # Get entries that have BOTH Price5Min (for selection) and BSP (for settlement)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP, ge.Price5Min, ge.PrizeMoney, ge.Weight,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-10-01'
    AND ge.Price5Min > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # Feature Engineering (Fast)
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0) # MATCH LIVE LOGIC
    df['StartPrice'] = df['Price5Min'] # For Lago Features, arguably could be either, keeping simple.
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    cols_to_lag = ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']
    for col in cols_to_lag:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)


    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['Margin_avg'] = df.groupby('GreyhoundID')['Margin1'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime'].replace(0, np.inf)
    df['RunSpeed_avg'] = df.groupby('GreyhoundID')['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
        
    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # PREDICT
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    # -------------------------------------------------------------
    # CRITICAL LOGIC: 
    # Decision Edge = Prob - (1 / Price5Min)
    # Settlement Price = BSP
    # -------------------------------------------------------------
    df['Implied_Prob_Decision'] = 1 / df['Price5Min']
    df['Edge_Decision'] = df['Prob_V33'] - df['Implied_Prob_Decision']
    
    # FILTER TO 2025 ONLY FOR BACKTESTING (But keep 2024 for feature calc)
    df = df[df['date_dt'] >= '2025-01-01'].copy()
    
    # BACK STRATEGY
    back_mask = (df['Box'].isin(BACK_BOXES)) & \
                (df['Price5Min'] <= BACK_MAX_ODDS) & \
                (df['Prob_V33'] > BACK_MIN_PROB) & \
                (df['Edge_Decision'] > BACK_MIN_EDGE)
    
    # LAY STRATEGY
    lay_mask = (df['Box'].isin(LAY_BOXES)) & \
               (df['Price5Min'] < LAY_MAX_PRICE) & \
               (df['Prob_V33'] < LAY_MAX_PROB) & \
               (df['Edge_Decision'] < LAY_MAX_EDGE) & \
               (~df['Track'].isin(STRAIGHT_TRACKS)) & \
               (df['Place_Lag2'] != -1)

    print("\n--- BET VOLUME ANALYSIS ---")
    print(f"Total Rows with Price5Min: {len(df)}")
    print(f"BACK Qualifiers (using Price5Min): {back_mask.sum()}")
    print(f"LAY Qualifiers (using Price5Min): {lay_mask.sum()}")
    
    # P/L CALCULATION (Using BSP)
    b_bets = df[back_mask].copy()
    if len(b_bets) > 0:
        target = BANKROLL * 0.06
        # StartPrice here is Price5Min (from feature eng). We need BSP for execution.
        b_bets['Stake'] = target / (b_bets['Price5Min'] - 1) # User script calc stake at time of bet
        # Warning: If BSP is lower, stake might be 'wrong' relative to risk, but we committed $X stake.
        # Actually for target profit, stake varies. Let's assume we bet fixed stake calculated at trigger.
        
        b_bets['PL'] = np.where(b_bets['win']==1, b_bets['Stake'] * (b_bets['BSP'] - 1) * (1-COMMISSION), -b_bets['Stake'])
        print(f"\n[BACK] P/L (Settle BSP): ${b_bets['PL'].sum():.2f} | ROI: {(b_bets['PL'].sum()/b_bets['Stake'].sum())*100:.2f}%")
        
    l_bets = df[lay_mask].copy()
    if len(l_bets) > 0:
        target = BANKROLL * 0.06
        # Lay Staking: usually calculated on Liability. 
        # liability = bank * 0.06. 
        # stake = liability / (price - 1). 
        # price = Price5Min (locked in? User said "betting at bsp"). 
        # IF BETTING AT BSP: Stake is calculated at BSP? Or Stake fixed?
        # User said "betting at bsp". For Back, BSP is standard. For Lay, BSP Liability is standard.
        # Let's assume Liability is fixed at $12 (6% of $200).
        liability = target
        
        # P/L for Lay at BSP:
        # If Win (Dog wins) -> Lose Liability
        # If Lose (Dog loses) -> Win Stake = Liability / (BSP - 1) * (1-Comm)
        
        l_bets['PL'] = np.where(l_bets['win']==1, 
                                -liability, 
                                (liability / (l_bets['BSP'] - 1)) * (1 - COMMISSION))
                                
        # Filter: Can't lay if BSP < 1.01 (Protect div zero)
        l_bets = l_bets[l_bets['BSP'] > 1.01]
        
        print(f"\n[LAY] P/L (Settle BSP): ${l_bets['PL'].sum():.2f} | Roi: {(l_bets['PL'].sum() / (len(l_bets)*liability))*100:.2f}%")

if __name__ == "__main__":
    run_hybrid_backtest()
