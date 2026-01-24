
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

INITIAL_BANK = 200.0
COMMISSION = 0.08  # User requested 8%

def get_data_and_features():
    print("[1/3] Loading Data & V33 Features...")
    conn = sqlite3.connect(DB_PATH)
    
    # Same query as before
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Price5Min, ge.PrizeMoney, ge.Weight,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.FinishTime > 0
    AND ge.BSP > 0
    ORDER BY rm.MeetingDate, r.RaceTime
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
    
    # CRITICAL FIX: CLEAN MARGIN1
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
    
    # Re-sort chronologically for backtest
    df = df.sort_values(['date_dt', 'RaceID'])
    return df, feature_cols

def run_back_target_sim(df, target_pct):
    bank = INITIAL_BANK
    bets = 0
    wins = 0
    turnover = 0
    max_drawdown = 0
    peak = INITIAL_BANK
    
    history = []
    
    for _, row in df.iterrows():
        prob = row['Prob_V33']
        price = row['StartPrice']
        if price <= 1.01: continue
        
        # Back Filter: Box 1-3, Odds <= 15, Prob > 0.55, Edge > 0.20
        implied = 1/price
        # Back Filter: Box 1-3, Odds <= 15, Prob > 0.55, Edge > 0.20
        implied = 1/price
        edge = prob - implied
        
        if row['Box'] in [1, 2, 3] and price <= 15.0 and prob > 0.55 and edge > 0.20:
            
            # TARGET PROFIT PERCENTAGE
            # Target = Bank * (target_pct / 100)
            # Stake = Target / (Odds - 1)
            
            target = bank * (target_pct / 100.0)
            stake = target / (price - 1)
            
            stake = max(stake, 0)
            stake = min(stake, bank) # Cannot bet more than bank
            
            if stake <= 0: continue
            
            bets += 1
            turnover += stake
            
            if row['Place'] == 1:
                profit = stake * (price - 1) * (1 - COMMISSION)
                bank += profit
                wins += 1
            else:
                bank -= stake
                
            if bank > peak: peak = bank
            dd = (peak - bank) / peak * 100
            if dd > max_drawdown: max_drawdown = dd
                
            history.append(bank)
            
    return {
        'Bank': bank, 'Bets': bets, 'Wins': wins, 'Turnover': turnover, 'DD': max_drawdown
    }

def main():
    df, feature_cols = get_data_and_features()
    
    print(f"[2/3] Predicting...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    print("\n" + "="*100)
    print("BACK STRATEGY: TARGET PROFIT RANGE ANALYSIS (2-10%)")
    print(f"Commission: {COMMISSION*100}% | Initial Bank: ${INITIAL_BANK}")
    print("="*100)
    
    print(f"{'TARGET %':<10} | {'BETS':<6} | {'STRIKE':<8} | {'PROFIT':<10} | {'ROI':<8} | {'FINAL BANK':<12} | {'MAX DD'}")
    print("-" * 100)
    
    # Range 2% to 10%
    percentages = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for pct in percentages:
        res = run_back_target_sim(df.copy(), pct)
        profit = res['Bank'] - INITIAL_BANK
        roi = (profit / res['Turnover'] * 100) if res['Turnover'] else 0
        strike_rate = (res['Wins'] / res['Bets'] * 100) if res['Bets'] else 0
        
        print(f"{pct:<2}% Bank   | {res['Bets']:<6} | {strike_rate:<7.1f}% | ${profit:<9.2f} | {roi:<7.2f}% | ${res['Bank']:<11.2f} | {res['DD']:.1f}%")

    print("-" * 100)

if __name__ == "__main__":
    main()
