"""
Backtest Value Betting Strategy (Probability Model)
Objective: Test if betting when Edge > 0 yields profit.
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

DB_PATH = 'greyhound_racing.db'
PROB_MODEL_PATH = 'models/prob_xgb_model.pkl'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    print("Loading test data (2024-2025)...")
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split,
        ge.FinishTime,
        ge.Position,
        ge.StartingPrice,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND ge.Split IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    # Parse Odds
    def parse_price(x):
        try:
            if not x or x is None: return np.nan
            x = str(x).replace('$', '').strip()
            if 'F' in x: x = x.replace('F', '')
            return float(x)
        except:
            return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Calculating Features...")
    # Same logic as training
    
    # Global Medians
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    # Rolling Stats
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    
    return df

def run_backtest(df):
    print("Running Backtest...")
    
    # Load Model
    with open(PROB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # Prepare X
    df_clean = df.dropna(subset=['s_Roll5', 'p_Roll5', 'Odds']).copy() # Must have history and odds
    features = ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    X = df_clean[features]
    
    # Predict Probability
    probs = model.predict_proba(X)[:, 1]
    df_clean['WinProb'] = probs
    df_clean['ModelPrice'] = 1 / df_clean['WinProb']
    
    # CALC EDGE
    # Edge = (Probability * Odds) - 1
    # e.g. Prob 0.50, Odds 2.50 -> (0.5 * 2.5) - 1 = 0.25 (25% Edge)
    df_clean['Edge'] = (df_clean['WinProb'] * df_clean['Odds']) - 1
    
    # Strategy Filters
    # 1. Min Probability (Don't bet on 100/1 shots even if model says 80/1)
    #    Let's only bet on dogs with > 15% chance ($6.50)
    df_clean['Viable'] = df_clean['WinProb'] >= 0.15
    
    # 2. Positive Edge
    edges = [0.0, 0.05, 0.10, 0.20] # 0%, 5%, 10%, 20% Edge
    
    print("\n" + "="*80)
    print("VALUE BETTING RESULTS (2024-2025)")
    print("Strategy: Bet if Edge > Threshold AND WinProb >= 15%")
    print("="*80)
    print(f"{'Edge >':<8} | {'Bets':<6} | {'Winners':<7} | {'Strike %':<8} | {'P/L':<8} | {'ROI %':<8}")
    print("-" * 80)
    
    for edge in edges:
        subset = df_clean[df_clean['Viable'] & (df_clean['Edge'] > edge)]
        
        if len(subset) == 0:
            print(f"{edge:<8} | 0      | 0       | 0.0%     | 0.00     | 0.0%")
            continue
            
        bets = len(subset)
        wins = subset[subset['Position'] == '1'].shape[0]
        strike = (wins / bets) * 100
        
        # Profit (Flat stake)
        profit = subset[subset['Position'] == '1']['Odds'].sum() - bets
        roi = (profit / bets) * 100
        
        print(f"{edge*100:3.0f}%     | {bets:<6} | {wins:<7} | {strike:<8.1f} | {profit:<8.1f} | {roi:<8.1f}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    run_backtest(df)
