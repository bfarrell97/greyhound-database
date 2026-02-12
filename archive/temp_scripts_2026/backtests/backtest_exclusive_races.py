
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
COMMISSION = 0.05 # Using 5% for backtest standard (or 0.10 if user insists?)
# predict_v33_tips uses 0.10 for LAY commissions but BACK is gross win?
# Actually, standard Betfair commission is applied on Net Winnings.
# Let's use 0.05 (5%) as it's standard. predict_v33_tips used 0.10 for Lay Liability calcs?
COMMISSION = 0.05

# CRITERIA FROM predict_v33_tips.py
MIN_PROB = 0.55
# MIN_EDGE = 0.20 # Changed to 0.10 to see if volume increases? No, user said "current betting strategy"
MIN_EDGE = 0.20 
ODDS_CAP = 15.0
# BOXES = [2, 3, 4, 5, 6, 7] # As seen in code (Line 294)

def get_data_and_features():
    print("[1/4] Loading Data & Features...")
    conn = sqlite3.connect(DB_PATH)
    
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
    
    # PRESERVE RAW BOX FOR FILTERING
    df['RawBox'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    
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

def run_simulation(df):
    bank = INITIAL_BANK
    history = []
    
    wins = 0
    bets_placed = 0
    turnover = 0
    
    print("[3/4] Filtering Candidates (Current Strategy)...")
    
    # Vectorized Candidate Identification
    # Criteria: Box 2-7, Price <= 15, Prob > 0.55, Edge > 0.20
    
    # Edge = Prob - (1/Price)
    implied_probs = 1.0 / df['StartPrice']
    edges = df['Prob_V33'] - implied_probs
    
    # Mask
    # Box check: RawBox in [2,3,4,5,6,7]
    box_mask = df['RawBox'].isin([2, 3, 4, 5, 6, 7])
    price_mask = df['StartPrice'] <= ODDS_CAP
    prob_mask = df['Prob_V33'] > MIN_PROB
    edge_mask = edges > MIN_EDGE
    
    candidates = df[box_mask & price_mask & prob_mask & edge_mask].copy()
    
    print(f"Found {len(candidates)} initial candidates.")
    
    print("[4/4] Applying Exclusion Logic (Max 2 Runners per Race)...")
    
    # Group by RaceID to count qualifiers
    race_counts = candidates.groupby('RaceID').size()
    
    # Identify Races to Exclude (Count > 2)
    excluded_races = race_counts[race_counts > 2].index
    
    # Filter out excluded races
    final_bets = candidates[~candidates['RaceID'].isin(excluded_races)].copy()
    
    excluded_count = len(candidates) - len(final_bets)
    print(f"Excluded {len(excluded_races)} races ({excluded_count} bets) due to > 2 qualifiers.")
    print(f"Final Bets to Simulate: {len(final_bets)}")
    
    print("-" * 100)
    print("Simulating P/L...")
    
    # Simulation Loop
    for _, row in final_bets.iterrows():
        price = row['StartPrice']
        win = (row['Place'] == 1)
        
        # TARGET PROFIT 6% STAKING
        # Stake = Target / (Price - 1)
        target_profit = 200.0 * 0.06 # Fixed Bankroll Basis ($12.00)
        # OR should it be live bank? predict_v33_tips uses fixed BANKROLL=200 constant.
        # User request: "current betting strategy and staking plan".
        # Current script uses fixed $200 basis.
        
        stake = target_profit / (price - 1)
        
        # Safety: Don't bet if price is crazy low (e.g. $1.01 -> stake $1200)
        # Though Price > 1.0 check exists.
        
        if win:
            profit = target_profit # Gross profit matches target exactly
            pnl = profit * (1 - COMMISSION)
            wins += 1
        else:
            pnl = -stake
            
        bank += pnl
        bets_placed += 1
        turnover += stake
        history.append(bank)
        
    return {
        'Final Bank': bank,
        'Profit': bank - INITIAL_BANK,
        'Bets': bets_placed,
        'Wins': wins,
        'Strike Rate': (wins/bets_placed*100) if bets_placed else 0,
        'ROI': ((bank - INITIAL_BANK) / turnover * 100) if turnover else 0,
        'Turnover': turnover
    }

def main():
    print("Backtest: V33 Strategy with Exclusion Rule")
    print("Rule: Exclude races with > 2 qualifiers.")
    print("Strategy: Box 2-7, Prob>0.55, Edge>0.20, Odds<=15")
    print("Staking: Target $12 Profit (6% of $200)")
    print("------------------------------------------------")
    
    df, feature_cols = get_data_and_features()
    
    print(f"[2/4] Predicting on {len(df)} entries...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    res = run_simulation(df)
    
    print("\nRESULTS")
    print("=" * 30)
    print(f"Bets Placed: {res['Bets']}")
    print(f"Wins:        {res['Wins']}")
    print(f"Strike Rate: {res['Strike Rate']:.2f}%")
    print(f"Turnover:    ${res['Turnover']:.2f}")
    print(f"Net Profit:  ${res['Profit']:.2f}")
    print(f"ROI:         {res['ROI']:.2f}%")
    print(f"Final Bank:  ${res['Final Bank']:.2f}")
    print("=" * 30)

if __name__ == "__main__":
    main()
