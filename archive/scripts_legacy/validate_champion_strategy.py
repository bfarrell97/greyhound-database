"""
Champion Strategy Validation Suite
Rigorous testing of the "Short Course Dominant" Strategy (Pace Gap > 0.15, Dist < 400, Prize > 20k).
Includes:
1. Walk-Forward / Quarterly Stability
2. Parameter Sensitivity (Heatmap)
3. Monte Carlo Permutation Test (Significance)
4. Monte Carlo Drawdown Simulation (Risk)
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def load_and_prep():
    print("Loading Data (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
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
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    print("Feature Engineering...")
    # Benchmarks
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    df = df.dropna(subset=['p_Roll5', 'Odds']).copy()
    
    print("Predicting...")
    with open(PACE_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    features = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    X = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    df['PredNormPace'] = model.predict(X)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    print("Calculating Gaps...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    return df

def get_trades(df, gap_thresh=0.15, odds_thresh=2.0, prize_thresh=20000, dist_thresh=400):
    candidates = df[df['Rank'] == 1].copy()
    trades = candidates[
        (candidates['Distance'] < dist_thresh) &
        (candidates['Gap'] >= gap_thresh) &
        (candidates['CareerPrize'] >= prize_thresh) &
        (candidates['Odds'] >= odds_thresh) &
        (candidates['Odds'] <= 30)
    ].copy()
    
    trades['Profit'] = trades.apply(lambda x: (x['Odds'] - 1) if x['Position'] == '1' else -1, axis=1)
    return trades

def run_validation():
    df = load_and_prep()
    
    # Base Trades
    trades = get_trades(df)
    
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE (2024-2025)")
    print("="*80)
    bats = len(trades)
    wins = len(trades[trades['Position'] == '1'])
    roi = (trades['Profit'].sum() / bats) * 100
    print(f"Bets: {bats}, Wins: {wins}, Strike: {(wins/bats)*100:.1f}%, ROI: {roi:.1f}%")
    
    # 1. Quarterly Stability
    print("\n" + "-"*80)
    print("1. QUARTERLY STABILITY")
    print("-"*80)
    trades['Quarter'] = trades['MeetingDate'].dt.to_period('Q')
    q_stats = trades.groupby('Quarter').agg(
        Bets=('Profit', 'count'),
        ROI=('Profit', lambda x: (x.sum() / len(x)) * 100),
        Strike=('Position', lambda x: (x == '1').mean() * 100)
    )
    print(q_stats)
    
    # 2. Parameter Sensitivity
    print("\n" + "-"*80)
    print("2. PARAMETER SENSITIVITY (Gap vs Odds)")
    print("-"*80)
    gaps = [0.10, 0.12, 0.15, 0.18, 0.20]
    odds = [1.50, 1.80, 2.00, 2.20, 2.50]
    
    print(f"{'Gap':<6} | " + " | ".join([f"Odds {o:<4}" for o in odds]))
    for g in gaps:
        row_str = f"{g:<6} | "
        for o in odds:
            t = get_trades(df, gap_thresh=g, odds_thresh=o)
            if len(t) < 50:
                roi_str = "N/A"
            else:
                r = (t['Profit'].sum() / len(t)) * 100
                roi_str = f"{r:5.1f}%"
            row_str += f"{roi_str:<9} | "
        print(row_str)

    # 3. Monte Carlo Permutation (Luck Test)
    print("\n" + "-"*80)
    print("3. MONTE CARLO PERMUTATION TEST (1000 runs)")
    print("Checking probability that ROI is random luck by shuffling race winners.")
    print("-"*80)
    
    # Filter to only the races we bet on
    race_keys = trades['RaceKey'].unique()
    relevant_races = df[df['RaceKey'].isin(race_keys)].copy()
    
    # We simulate random winners in these races 1000 times and check strategy ROI
    # Faster approx: Shuffle 'IsWin' within the Strategy Trades? 
    # No, that ignores that we selected specific dogs.
    # Proper Permutation: Shuffle the "Position" column relative to "RaceKey"? 
    # Too complex/slow for script.
    # Standard Permutation: Null Hypothesis = Strategy has 0 edge. 
    # Sample outcomes from the *market average* win rate implied by odds?
    # Let's do a simple shuffle of the Trade Outcomes relative to Market Odds.
    # If we bet on dogs with avg odds $2.80, implied win rate is 35%.
    # Let's simulate 1000 separate equity curves using the *Market Implied Probability* for each bet.
    # If our Actual Curve is > 95% of Random Curves, it's skill.
    
    n_sims = 1000
    actual_profit = trades['Profit'].sum()
    better_runs = 0
    
    # Avg implied prob
    trades['ImpliedProb'] = 1 / trades['Odds']
    
    for i in range(n_sims):
        # Simulate W/L based on Implied Prob
        random_outcomes = np.random.rand(len(trades)) < trades['ImpliedProb']
        # Calc Profit: Win = Odds-1, Loss = -1
        sim_profit = np.where(random_outcomes, trades['Odds'] - 1, -1).sum()
        if sim_profit >= actual_profit:
            better_runs += 1
            
    p_value = better_runs / n_sims
    print(f"Actual Profit: {actual_profit:.1f}u")
    print(f"P-Value (Chance result is luck): {p_value:.4f}")
    if p_value < 0.05:
        print("RESULT: Statistically Significant (Skill > Luck)")
    else:
        print("RESULT: Not Significant (Could be Luck)")

    # 4. Drawdown Simulation
    print("\n" + "-"*80)
    print("4. DRAWDOWN STRESS TEST (Shuffle Order)")
    print("-"*80)
    
    max_dds = []
    profits = trades['Profit'].values
    
    for i in range(1000):
        np.random.shuffle(profits)
        equity = np.cumsum(profits)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dds.append(dd.max())
        
    avg_dd = np.mean(max_dds)
    worst_dd = np.max(max_dds)
    print(f"Average Max Drawdown: {avg_dd:.1f}u")
    print(f"Worst Case Drawdown:  {worst_dd:.1f}u")
    print(f"Rec. Bankroll (2x Worst DD): {worst_dd*2:.1f}u")

if __name__ == "__main__":
    run_validation()
