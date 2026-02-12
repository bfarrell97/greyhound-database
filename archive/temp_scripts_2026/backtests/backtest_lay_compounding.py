
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
import re
from datetime import datetime

# Pathing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# CONFIG
MODEL_PATH = "models/xgb_v33_prod.json"
START_BANK = 200.0
TARGET_PCT = 0.06
COMMISSION = 0.05 # Betfair standard
# LAY Criteria from predict_v33_tips.py
MIN_PRICE = 1.06 # Exclude 1.05 placeholders
MAX_PRICE = 2.00
MAX_PROB = 0.20
MAX_EDGE = -0.48 # User requested fine-tuning (was -0.45)

# STRAIGHT TRACKS FILTER
STRAIGHT_TRACKS = [
    'Healesville', 
    'Murray Bridge (MBS)', 
    'Richmond (RIS)', 
    'Richmond Straight', 
    'Capalaba', 
    'Q Straight'
]

def get_v33_features(df):
    """Replicate V33 Feature Engineering"""
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    if 'Margin1' in df.columns:
        df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    
    for col in ['Place', 'StartPrice', 'RunTime', 'SplitMargin', 'Distance']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0)

    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)
            
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['Margin_avg'] = df.groupby('GreyhoundID')['Margin1'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime']
    df['RunSpeed'] = df['RunSpeed'].replace([np.inf, -np.inf], 0)
    df['RunSpeed_avg'] = df.groupby('GreyhoundID')['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    
    df['Box'] = df['RawBox']
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
        
    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df, feature_cols

def run_backtest():
    print("="*60)
    print(f"V33 LAY STRATEGY - COMPOUNDING BACKTEST (PRODUCTION SETTINGS)")
    print(f"Criteria: Box 1&2, BSP < $2.00, Prob < 0.20, Edge < -0.48, Circle Tracks Only")
    print(f"Staking: Target {TARGET_PCT*100}% Profit of Live Bank")
    print("="*60)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # 1. Fetch Data (2024 only for speed/relevance)
    print("Fetching 2024 data...")
    query = """
    SELECT 
        ge.GreyhoundID, ge.Box as RawBox, ge.Position as Place, 
        ge.FinishTime as RunTime, ge.Split as SplitMargin, ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Weight, r.Distance, r.Grade, 
        t.TrackName as Track, rm.MeetingDate as date_dt,
        rm.MeetingID, r.RaceID, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.BSP > 0
    ORDER BY rm.MeetingDate ASC, r.RaceNumber ASC
    """
    df = pd.read_sql_query(query, conn)
    df['date_dt'] = pd.to_datetime(df['date_dt'])

    # FILTER FOR CIRCLE TRACKS ONLY (Exclude Straight)
    print(f"Loaded {len(df)} rows (All Tracks).")
    df = df[~df['Track'].isin(STRAIGHT_TRACKS)].copy()
    print(f"Filtered to {len(df)} rows (Circle Tracks Only).")
    
    if df.empty:
        print("No straight track data found! Check track names.")
        return
    
    print(f"Loaded {len(df)} rows.")
    
    # 2. Feature Engineering
    print("Engineering features...")
    processed, feature_cols = get_v33_features(df)
    
    # 3. Predict matches
    print("Loading Model...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    print("Predicting...")
    # Fix ALL infinite values
    processed = processed.replace([np.inf, -np.inf], 0)
    
    # Batch predict
    dmatrix = xgb.DMatrix(processed[feature_cols])
    processed['ModelProb'] = model.predict(dmatrix)
    
    # 4. Simulation
    print("\nRunning Simulation...")
    
    bank = START_BANK
    high_water_mark = START_BANK
    max_drawdown = 0.0
    
    bets_placed = 0
    wins = 0
    losses = 0
    
    # Analyze day by day to simulate bank updates properly
    processed['Date'] = processed['date_dt'].dt.date
    daily_groups = processed.groupby('Date')
    
    history_equity = []
    
    total_liability_risk = 0.0
    box_stats = {}
    
    for date, group in daily_groups:
        # Filter Candidates
        # Edge Calculation (ABSOLUTE: Prob - Implied)
        # Implied = 1 / Price
        # Edge = Prob - Implied
        # We want Edge < -0.50
        
        candidates = group[
            (group['RawBox'].isin([1, 2])) & # Production: Box 1 & 2 Sniper
            (group['StartPrice'] < MAX_PRICE) & 
            (group['StartPrice'] >= MIN_PRICE) &
            (group['ModelProb'] < MAX_PROB) &
            (group['ModelProb'] - (1/group['StartPrice']) < MAX_EDGE) &
            (group['Place_Lag2'] != -1) # Has History
        ].copy()
        
        if candidates.empty:
            continue
            
        # Sort by time/race (proxy via index/RaceNumber)
        candidates = candidates.sort_values(['RaceID'])
        
        for idx, row in candidates.iterrows():
            price = row['StartPrice']
            box = int(row['RawBox'])
            
            # STAKING
            # Target Profit = 6% of CURRENT Bank
            target_profit = bank * TARGET_PCT
            
            # Backer's Stake (S) needed to net target_profit
            # Profit = S * (1 - Comm)
            # S = Profit / (1 - Comm)
            stake_s = target_profit / (1 - COMMISSION)
            
            # Liability = S * (Price - 1)
            liability = stake_s * (price - 1)
            
            # Safety Check: Can we afford liability?
            if liability > bank:
                # Cap the bet to bank
                # Liability = Bank
                # Bank = S * (Price - 1) -> S = Bank / (Price - 1)
                stake_s = bank / (price - 1)
                liability = bank
            
            total_liability_risk += liability
            bets_placed += 1
            
            # RESULT
            # We are LAYING. We WIN if the dog LOSES (Place != 1).
            # We LOSE if the dog WINS (Place == 1).
            
            is_winner = (row['Place'] != 1)
            
            if is_winner:
                # We Keep the Backer's Stake (minus comm)
                profit = stake_s * (1 - COMMISSION)
                bank += profit
                wins += 1
                pnl = profit
            else:
                # We Pay Out Liability
                bank -= liability
                losses += 1
                pnl = -liability
            
            # Track Box Stats
            if box not in box_stats:
                box_stats[box] = {'Bets': 0, 'Wins': 0, 'Profit': 0.0}
            box_stats[box]['Bets'] += 1
            if is_winner: box_stats[box]['Wins'] += 1
            box_stats[box]['Profit'] += pnl

            # Drawdown Tracking
            if bank > high_water_mark:
                high_water_mark = bank
            
            dd = (high_water_mark - bank) / high_water_mark
            if dd > max_drawdown:
                max_drawdown = dd
                
        history_equity.append({'Date': date, 'Bank': bank, 'DD': max_drawdown})
        
    print("\n" + "="*40)
    print("RESULTS (2024-Present)")
    print("="*40)
    print(f"Start Bank:     ${START_BANK:.2f}")
    print(f"Final Bank:     ${bank:,.2f}")
    print(f"Total Return:   {((bank - START_BANK)/START_BANK)*100:+.1f}%")
    print(f"Max Drawdown:   {max_drawdown*100:.1f}%")
    print("-" * 40)
    print(f"Total Bets:     {bets_placed}")
    print(f"Win Rate:       {wins/bets_placed*100:.1f}%" if bets_placed else "Win Rate: 0%")
    print(f"Avg Liability:  ${total_liability_risk/bets_placed:.2f}" if bets_placed else "Avg Liab: $0")
    print("="*40)
    
    print("\nBOX ANALYSIS (Lay Strategy)")
    print("-" * 60)
    print(f"{'Box':<5} | {'Bets':<6} | {'LayWins':<8} | {'Strike%':<8} | {'Profit':<10}")
    print("-" * 60)
    for box in range(1, 9):
        if box in box_stats:
            stats = box_stats[box]
            bets = stats['Bets']
            lwins = stats['Wins']
            profit = stats['Profit']
            strike = (lwins / bets * 100) if bets > 0 else 0
            print(f"{box:<5} | {bets:<6} | {lwins:<8} | {strike:>6.1f}%  | ${profit:>9.2f}")
        else:
            print(f"{box:<5} | 0      | 0        | 0.0%     | $0.00")
    print("-" * 60)
    
    conn.close()

if __name__ == "__main__":
    run_backtest()
