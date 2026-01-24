
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_test_2020_2024.json" # Back to test model

# FILTERS
LAY_MAX_PROB = 0.20
LAY_MAX_EDGE = -0.48
LAY_MAX_PRICE = 2.00
LAY_BOXES = [1, 2]
STRAIGHT_TRACKS = ['Healesville', 'Murray Bridge (MBS)', 'Richmond (RIS)', 'Richmond Straight', 'Capalaba', 'Q Straight']

def run_backtest_price5min():
    print("="*80)
    print("V33 BACKTEST: USING PRICE_5MIN (Simulating Live)")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
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
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.FinishTime > 0
    AND (ge.BSP > 0 OR ge.Price5Min > 0)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Feature Engineering (Fast Version)
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # USE PRICE 5MIN AS START PRICE IF AVAILABLE, ELSE BSP
    df['StartPrice'] = np.where(df['Price5Min'] > 0, df['Price5Min'], df['BSP'])
    df['StartPrice'] = pd.to_numeric(df['StartPrice'], errors='coerce').fillna(0)
    df = df[df['StartPrice'] > 1.0].copy()
    
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(0)
    
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    # Lags
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
    
    # PREDICT
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    df['Implied_Prob'] = 1 / df['StartPrice']
    df['Edge'] = df['Prob_V33'] - df['Implied_Prob']

    # FILTER
    mask = (df['Box'].isin(LAY_BOXES)) & \
           (df['StartPrice'] < LAY_MAX_PRICE) & \
           (df['Prob_V33'] < LAY_MAX_PROB) & \
           (df['Edge'] < LAY_MAX_EDGE) & \
           (~df['Track'].isin(STRAIGHT_TRACKS)) & \
           (df['Place_Lag2'] != -1)
           
    bets = df[mask].copy()
    
    print(f"BETS FOUND: {len(bets)}")
    if len(bets) > 0:
        # P/L using BSP for settlement (Live betting places Fixed Price usually, checking manual)
        # Manual says: "Bet Execution: Current lay price (NOT BSP) - locked in at bet time (< $2.00)"
        # So we settle at the Price we took (StartPrice here is Price5Min)
        
        bets['Lay_Result'] = np.where(bets['Place'] > 1, 'WIN', 'LOSS')
        
        # Stake/Risk
        COMMISSION = 0.10
        BANKROLL = 200.0
        TARGET = BANKROLL * 0.06
        
        bets['Lay_Stake'] = TARGET / (bets['StartPrice'] - 1)
        bets['Risk'] = TARGET
        
        bets['PL'] = np.where(
            bets['Lay_Result'] == 'WIN',
            bets['Lay_Stake'] * (1 - COMMISSION),
            -bets['Risk']
        )
        
        profit = bets['PL'].sum()
        risk = bets['Risk'].sum()
        roi = (profit/risk)*100
        print(f"P/L: ${profit:.2f} | ROI: {roi:.2f}%")
        
        # Bets per month
        bets['Month'] = bets['date_dt'].dt.to_period('M')
        print("\nBets per Month:")
        print(bets.groupby('Month').size())

if __name__ == "__main__":
    run_backtest_price5min()
