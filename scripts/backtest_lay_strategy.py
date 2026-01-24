
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_prod.json"

STAKE_SIZE = 10.0 
COMMISSION = 0.10

# FILTERS
MAX_PROB = 0.20   # Model thinks chance is < 20%
MAX_EDGE = -0.50  # (ModelProb - ImpliedProb) < -50%
MAX_PRICE = 2.00  # Implied by Edge -0.50 anyway

def get_data_and_features():
    print("[1/3] Loading Data & V33 Features...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Price5Min, ge.PrizeMoney, ge.Weight,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt,
        (SELECT COUNT(*) FROM GreyhoundEntries ge2 WHERE ge2.RaceID = ge.RaceID) as FieldSize
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.FinishTime > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Preprocessing
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    df['StartPrice'] = pd.to_numeric(df['StartPrice'], errors='coerce').fillna(0)
    df = df[df['StartPrice'] > 1.0].copy()
    
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(0)
    
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    # Features (Standard V33 Lags)
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)
            
    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['Margin_avg'] = df.groupby('GreyhoundID')['Margin1'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime']
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
    return df, feature_cols

def run_lay_backtest():
    print("="*80)
    print(f"LAY BETTING BACKTEST: Edge < {MAX_EDGE} | Comm {COMMISSION*100}%")
    print("Testing Staking Methods...")
    print("="*80)
    
    df, feature_cols = get_data_and_features()
    
    # PREDICT
    print(f"[2/3] Predicting on {len(df)} entries...")
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        dtest = xgb.DMatrix(df[feature_cols])
        df['Prob_V33'] = model.predict(dtest)
        df['Implied_Prob'] = 1 / df['StartPrice']
        # Use ABSOLUTE edge (prob - implied)
        df['Edge'] = df['Prob_V33'] - df['Implied_Prob']
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # FILTER
    mask = (df['StartPrice'] <= MAX_PRICE) & \
           (df['Prob_V33'] < MAX_PROB) & \
           (df['Edge'] < MAX_EDGE)
    
    base_bets = df[mask].copy()
    print(f"\nIdentified {len(base_bets)} Lay Opportunities.")
    
    if len(base_bets) == 0:
        return

    # ITERATE STAKING
    METHODS = ['Fixed Liability ($10)', 'Fixed Stake ($10)', 'Target Profit ($10)', 'Kelly/16 (Lay)', 'Target Profit (4% Bank)']
    BANKROLL = 200.0
    
    print("\n" + "="*100)
    print(f"{'METHOD':<25} | {'BETS':<6} | {'STRIKE':<8} | {'PROFIT':<10} | {'ROI':<8} | {'DRAWDOWN':<8}")
    print("-" * 100)
    
    for method in METHODS:
        bets = base_bets.copy()
        
        if method == 'Fixed Liability ($10)':
            LIABILITY = 10.0
            bets['Lay_Stake'] = LIABILITY / (bets['StartPrice'] - 1)
            bets['Risk'] = LIABILITY
            
        elif method == 'Fixed Stake ($10)':
            STAKE = 10.0 
            bets['Lay_Stake'] = STAKE
            bets['Risk'] = STAKE * (bets['StartPrice'] - 1)
            
        elif method == 'Target Profit ($10)':
            TARGET = 10.0
            bets['Lay_Stake'] = TARGET / (1 - COMMISSION)
            bets['Risk'] = bets['Lay_Stake'] * (bets['StartPrice'] - 1)
            
        elif method == 'Kelly/16 (Lay)':
            # For Laying: Edge = Implied - Model = -df['Edge']
            bets['Lay_Edge'] = -bets['Edge']
            # Kelly fraction for liability: f = edge / (odds - 1)
            bets['kelly_frac'] = bets['Lay_Edge'] / (bets['StartPrice'] - 1)
            bets['kelly_frac'] = bets['kelly_frac'].clip(lower=0, upper=0.25)  # Safety Cap
            # Liability (our risk)
            bets['Risk'] = (bets['kelly_frac'] * BANKROLL) / 16
            # Lay Stake = Liability / (Odds - 1)
            bets['Lay_Stake'] = bets['Risk'] / (bets['StartPrice'] - 1)
            
        elif method == 'Target Profit (4% Bank)':
            # Target 4% of bankroll as profit on each win
            TARGET = BANKROLL * 0.04  # $8 on $200 bank
            bets['Lay_Stake'] = TARGET / (1 - COMMISSION)
            bets['Risk'] = bets['Lay_Stake'] * (bets['StartPrice'] - 1)

        # Calculate P/L
        bets['Lay_Result'] = np.where(bets['Place'] > 1, 'WIN', 'LOSS')
        
        bets['PL'] = np.where(
            bets['Lay_Result'] == 'WIN',
            bets['Lay_Stake'] * (1 - COMMISSION),
            -bets['Risk']
        )
        
        total_profit = bets['PL'].sum()
        total_risk_turnover = bets['Risk'].sum()
        roi = (total_profit / total_risk_turnover) * 100
        
        wins = len(bets[bets['Lay_Result'] == 'WIN'])
        strike_rate = (wins / len(bets)) * 100
        
        # DD
        bets['Bank'] = 200 + bets['PL'].cumsum()
        peak = bets['Bank'].cummax()
        dd = ((peak - bets['Bank']) / peak).max() * 100
        
        print(f"{method:<25} | {len(bets):<6} | {strike_rate:<7.1f}% | ${total_profit:<9.2f} | {roi:<7.2f}% | {dd:<7.1f}%")

    print("-" * 100)

if __name__ == "__main__":
    run_lay_backtest()
