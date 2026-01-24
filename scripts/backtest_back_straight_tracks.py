
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
COMMISSION = 0.05 

# CRITERIA FROM predict_v33_tips.py
MIN_PROB = 0.55
MIN_EDGE = 0.20 
ODDS_CAP = 15.0

# STRAIGHT TRACKS FILTER
STRAIGHT_TRACKS = [
    'Healesville', 
    'Murray Bridge (MBS)', 
    'Richmond (RIS)', 
    'Richmond Straight', 
    'Capalaba', 
    'Q Straight'
]

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
    
    # FILTER FOR STRAIGHT TRACKS
    print(f"Loaded {len(df)} rows (All Tracks).")
    df = df[df['Track'].isin(STRAIGHT_TRACKS)].copy()
    print(f"Filtered to {len(df)} rows (Straight Only).")
    
    if df.empty:
        print("No straight track data found!")
        return pd.DataFrame(), []

    # PRESERVE RAW BOX FOR FILTERING
    df['RawBox'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    
    # FIX: Margin handling (Match V33 parity)
    if 'Margin1' in df.columns:
        df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0)
    
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
    history = [INITIAL_BANK]
    peak_bank = INITIAL_BANK
    max_drawdown = 0.0
    
    wins = 0
    bets_placed = 0
    turnover = 0
    
    print("[3/4] Filtering Candidates (Current Strategy)...")
    
    # Vectorized Candidate Identification
    implied_probs = 1.0 / df['StartPrice']
    edges = df['Prob_V33'] - implied_probs
    
    # Mask
    box_mask = df['RawBox'].isin([2, 3, 4, 5, 6, 7]) # Strategic: Exclude 1 (Trap) & 8 (Wide)
    price_mask = df['StartPrice'] <= ODDS_CAP
    prob_mask = df['Prob_V33'] > MIN_PROB
    edge_mask = edges > MIN_EDGE
    
    candidates = df[box_mask & price_mask & prob_mask & edge_mask].copy()
    
    print(f"Found {len(candidates)} initial candidates.")
    
    print("[4/4] Applying Exclusion Logic (Max 2 Runners per Race)...")
    
    # Group by RaceID to count qualifiers
    race_counts = candidates.groupby('RaceID').size()
    excluded_races = race_counts[race_counts > 2].index
    final_bets = candidates[~candidates['RaceID'].isin(excluded_races)].copy()
    
    excluded_count = len(candidates) - len(final_bets)
    print(f"Excluded {len(excluded_races)} races ({excluded_count} bets) due to > 2 qualifiers.")
    print(f"Final Bets to Simulate: {len(final_bets)}")
    
    print("-" * 100)
    print("Simulating P/L (Compounding 6%)...")
    
    # Simulation Loop
    for _, row in final_bets.iterrows():
        price = row['StartPrice']
        win = (row['Place'] == 1)
        
        # COMPOUNDING STAKING
        # Target Profit = 6% of CURRENT Bankroll
        target_profit = bank * 0.06
        
        # Stake = Target / (Price - 1)
        stake = target_profit / (price - 1)
        
        # Safety: If stake > bank (impossible usually with 6% target unless price < 1.06), cap it
        if stake > bank: stake = bank
        
        if win:
            profit = target_profit
            pnl = profit * (1 - COMMISSION)
            wins += 1
        else:
            pnl = -stake
            
        bank += pnl
        bets_placed += 1
        turnover += stake
        history.append(bank)
        
        # Drawdown Calc
        if bank > peak_bank:
            peak_bank = bank
        
        dd = (peak_bank - bank) / peak_bank
        if dd > max_drawdown:
            max_drawdown = dd

        # DEBUG CHECK
        if bets_placed % 50 == 0:
            print(f"[DEBUG] Bet {bets_placed}: Wins={wins}, Bank={bank:.2f}")

        # Store bet result for Box Analysis

        # Store bet result for Box Analysis
        final_bets.at[_, 'Stake'] = stake
        final_bets.at[_, 'Profit'] = pnl
        final_bets.at[_, 'Win'] = 1 if win else 0

    print("-" * 100)
    print("BOX ANALYSIS (Straight Tracks)")
    print("-" * 100)
    print(f"{'Box':<5} | {'Bets':<6} | {'Wins':<5} | {'Strike%':<8} | {'Turnover':<10} | {'Profit':<10} | {'ROI%':<7}")
    print("-" * 80)
    
    box_stats = final_bets.groupby('RawBox').agg({
        'Profit': 'sum',
        'Stake': 'sum',
        'Win': 'sum',
        'GreyhoundID': 'count' # Count bets
    }).rename(columns={'GreyhoundID': 'Bets'})
    
    for box in range(1, 9):
        if box in box_stats.index:
            row = box_stats.loc[box]
            bets = int(row['Bets'])
            wins = int(row['Win'])
            stake = row['Stake']
            profit = row['Profit']
            strike = (wins / bets * 100) if bets > 0 else 0
            roi = (profit / stake * 100) if stake > 0 else 0
            
            print(f"{box:<5} | {bets:<6} | {wins:<5} | {strike:>6.1f}%  | ${stake:>9.2f} | ${profit:>9.2f} | {roi:>6.1f}%")
        else:
            print(f"{box:<5} | {'0':<6} | {'0':<5} | {'0.0%':>8} | {'$0.00':>10} | {'$0.00':>10} | {'0.0%':>7}")
            
    
    # Recalculate Wins from verifyed column data
    wins = int(final_bets['Win'].sum())

    return {
        'Final Bank': bank,
        'Profit': bank - INITIAL_BANK,
        'Bets': bets_placed,
        'Wins': wins,
        'Strike Rate': (wins/bets_placed*100) if bets_placed else 0,
        'ROI': ((bank - INITIAL_BANK) / turnover * 100) if turnover else 0,
        'Turnover': turnover,
        'Max Drawdown': max_drawdown * 100
    }

def main():
    print("Backtest: V33 BACK Strategy - STRAIGHT TRACKS ONLY")
    print(f"Tracks: {STRAIGHT_TRACKS}")
    print("Strategy: Box 2-7, Prob>0.55, Edge>0.20, Odds<=15")
    print("Staking: Target Profit 6% of CURRENT BANK (Compounding)")
    print("------------------------------------------------")
    
    df, feature_cols = get_data_and_features()
    
    if df.empty: return

    print(f"[2/4] Predicting on {len(df)} entries...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    res = run_simulation(df)
    
    print("\nRESULTS (Compounding)")
    print("=" * 30)
    print(f"Bets Placed:  {res['Bets']}")
    print(f"Wins:         {res['Wins']}")
    print(f"Strike Rate:  {res['Strike Rate']:.2f}%")
    print(f"Turnover:     ${res['Turnover']:.2f}")
    print(f"Net Profit:   ${res['Profit']:.2f}")
    print(f"ROI:          {res['ROI']:.2f}%")
    print(f"Max Drawdown: {res['Max Drawdown']:.2f}%")
    print(f"Final Bank:   ${res['Final Bank']:.2f} (from ${INITIAL_BANK})")
    print("=" * 30)

if __name__ == "__main__":
    main()
