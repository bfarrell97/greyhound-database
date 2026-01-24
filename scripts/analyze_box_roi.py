"""
Box-by-Box ROI Analysis with Target Profit 6% Staking
Tests each Box (1-8) individually to find optimal selection
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_prod.json"

INITIAL_BANK = 200.0
COMMISSION = 0.08
TARGET_PCT = 0.06  # 6% Target Profit

def get_data_and_features():
    print("[1/3] Loading Data & V33 Features...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, ge.Box as RawBox,
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
    
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    df['StartPrice'] = pd.to_numeric(df['StartPrice'], errors='coerce').fillna(0)
    df = df[df['StartPrice'] > 1.0].copy()
    
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0)
    
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
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
    df = df.sort_values(['date_dt', 'RaceID'])
    return df, feature_cols

def run_box_sim(df, box_filter):
    """Run simulation for specific box(es)"""
    bank = INITIAL_BANK
    bets = 0
    wins = 0
    turnover = 0
    max_dd = 0
    peak = INITIAL_BANK
    
    for _, row in df.iterrows():
        prob = row['Prob_V33']
        price = row['StartPrice']
        if price <= 1.01: continue
        
        implied = 1/price
        edge = prob - implied
        
        # Filter: Box, Odds <= 15, Prob > 0.55, Edge > 0.20
        if row['RawBox'] in box_filter and price <= 15.0 and prob > 0.55 and edge > 0.20:
            # Target Profit 6%
            target = bank * TARGET_PCT
            stake = target / (price - 1)
            stake = max(0, min(stake, bank))
            
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
            dd = (peak - bank) / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            
    return {'Bank': bank, 'Bets': bets, 'Wins': wins, 'Turnover': turnover, 'MaxDD': max_dd}

def run_lay_box_sim(df, box_filter):
    """Run LAY simulation for specific box(es)"""
    bank = INITIAL_BANK
    bets = 0
    wins = 0 # Successful Lays (dog lost)
    turnover = 0 # Liability risked
    max_dd = 0
    peak = INITIAL_BANK
    
    # Lay Target 4%
    TARGET_LAY_PCT = 0.04
    
    for _, row in df.iterrows():
        prob = row['Prob_V33']
        price = row['StartPrice']
        if price <= 1.01: continue
        
        implied = 1/price
        edge = prob - implied
        
        # Lay Filter: Price < 2.00, Prob < 0.20, Edge < -0.50 (Absolute)
        if row['RawBox'] in box_filter and price < 2.00 and prob < 0.20 and edge < -0.50:
            
            target = bank * TARGET_LAY_PCT
            stake = target / (1 - COMMISSION) # Stake to win target after comm
            liability = stake * (price - 1)
            
            # Check if we can afford liability
            if liability > bank:
                # Scale down
                ratio = bank / liability
                liability = bank
                stake = stake * ratio
            
            if stake <= 0: continue
            
            bets += 1
            turnover += liability # Liability is the risk in Laying
            
            if row['Place'] != 1: # Dog LOST = Lay WON
                profit = stake * (1 - COMMISSION)
                bank += profit
                wins += 1
            else: # Dog WON = Lay LOST
                bank -= liability
                
            if bank > peak: peak = bank
            dd = (peak - bank) / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            
    return {'Bank': bank, 'Bets': bets, 'Wins': wins, 'Turnover': turnover, 'MaxDD': max_dd}

def main():
    df, feature_cols = get_data_and_features()
    
    print(f"[2/3] Predicting...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    print("\n" + "="*100)
    print("BOX-BY-BOX BACK ANALYSIS (Target Profit 6%, Commission 8%)")
    print("="*100)
    
    print(f"\n{'BOX':<10} | {'BETS':<6} | {'WINS':<6} | {'STRIKE':<8} | {'PROFIT':<12} | {'ROI':<10} | {'MAX DD'}")
    print("-" * 100)
    
    # Test each individual box (BACK)
    for box in range(1, 9):
        res = run_box_sim(df.copy(), [box])
        profit = res['Bank'] - INITIAL_BANK
        roi = (profit / res['Turnover'] * 100) if res['Turnover'] > 0 else 0
        strike = (res['Wins'] / res['Bets'] * 100) if res['Bets'] > 0 else 0
        indicator = "✅" if roi > 0 else "❌"
        print(f"Box {box:<6} | {res['Bets']:<6} | {res['Wins']:<6} | {strike:<7.1f}% | ${profit:<11.2f} | {roi:<9.2f}% | {res['MaxDD']:.1f}% {indicator}")
        
    print("\n" + "="*100)
    print("BOX-BY-BOX LAY ANALYSIS (Target Profit 4%, Commission 8%)")
    print("="*100)
    print(f"\n{'BOX':<10} | {'BETS':<6} | {'WINS':<6} | {'STRIKE':<8} | {'PROFIT':<12} | {'ROI':<10} | {'MAX DD'}")
    print("-" * 100)

    # Test each individual box (LAY)
    for box in range(1, 9):
        res = run_lay_box_sim(df.copy(), [box])
        profit = res['Bank'] - INITIAL_BANK
        roi = (profit / res['Turnover'] * 100) if res['Turnover'] > 0 else 0
        strike = (res['Wins'] / res['Bets'] * 100) if res['Bets'] > 0 else 0
        indicator = "✅" if roi > 0 else "❌"
        print(f"Box {box:<6} | {res['Bets']:<6} | {res['Wins']:<6} | {strike:<7.1f}% | ${profit:<11.2f} | {roi:<9.2f}% | {res['MaxDD']:.1f}% {indicator}")

    print("-" * 100)
    
    # Test common combinations
    print("\nCOMBINATIONS (BACK):")
    print("-" * 100)
    
    combos = [
        ([1, 2, 3], "Box 1-3 (Inside)"),
        ([2, 3, 4], "Box 2-4 (Original Bug)"),
        ([2, 3, 4, 5, 6, 7], "Box 2-7 (Current)"),
        ([1, 8], "Box 1 & 8"),
    ]
    
    for boxes, name in combos:
        res = run_box_sim(df.copy(), boxes)
        profit = res['Bank'] - INITIAL_BANK
        roi = (profit / res['Turnover'] * 100) if res['Turnover'] > 0 else 0
        strike = (res['Wins'] / res['Bets'] * 100) if res['Bets'] > 0 else 0
        indicator = "✅" if roi > 0 else "❌"
        print(f"{name:<22} | {res['Bets']:<6} | {res['Wins']:<6} | {strike:<7.1f}% | ${profit:<11.2f} | {roi:<9.2f}% | {res['MaxDD']:.1f}% {indicator}")
    
    print("-" * 100)

if __name__ == "__main__":
    main()
