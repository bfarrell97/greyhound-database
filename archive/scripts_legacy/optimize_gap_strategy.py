"""
Gap / Dominance Strategy Optimizer
Test if betting on the Pace Leader is profitable ONLY when they have a significant predicted time advantage over the field.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def run_gap_analysis():
    conn = sqlite3.connect(DB_PATH)
    print("Loading Data...")
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
    
    # Feature Eng (Simplified to Pace Model needs)
    print("Feature Eng...")
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter
    df = df.dropna(subset=['p_Roll5', 'Odds']).copy()
    
    # Predict
    print("Predicting Pace...")
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    cols = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    X = df[cols].copy()
    X.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormPace'] = pace_model.predict(X)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    # RANKING & GAP CALCULATION
    print("Analyzing Gaps...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Filter valid races first (Min 6 dogs, Min prize etc for speed?)
    # Let's keep filters standard: Dist <= 600, Prize > 20k (found in optimizer), MinOdds > 1.50
    df = df[df['Distance'] <= 600]
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # To calc gap, we need the predictions of ALL dogs in the race
    # Sort by Race and PredPace
    df = df.sort_values(['RaceKey', 'PredPace'])
    
    # Calculate Rank
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    
    # Get 2nd Place Pace (shift -1 because sorted asc)
    # Actually, simpler:
    # 1. Pivot or Group to find 2nd best time
    # 2. Subtract Leader Time from 2nd Best Time
    
    # Group method
    g_race = df.groupby('RaceKey')
    df['LeaderTime'] = g_race['PredPace'].transform('first')
    df['SecondTime'] = g_race['PredPace'].transform(lambda x: x.nsmallest(2).iloc[-1] if len(x) >= 2 else np.nan)
    
    df['PredGap'] = df['SecondTime'] - df['LeaderTime']
    # If I am the leader, Gap is positive (Second - Me)
    # If I am not leader, Gap helps filter valid races?
    # We only care about betting the Leader.
    
    # Filter to Leaders
    leaders = df[df['Rank'] == 1].copy()
    
    # Optimization Loop for Gap Threshold
    gaps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    prizes = [20000] # Optimised previously
    min_odds = [1.50, 2.00]
    
    print("\n" + "="*80)
    print("DOMINANT LEADER STRATEGY RESULTS")
    print("="*80)
    print(f"{'Gap >':<8} | {'Odds >':<8} | {'Bets':<6} | {'Winners':<7} | {'Strike %':<8} | {'ROI %':<8}")
    self_results = []
    
    for gap in gaps:
        for odd in min_odds:
            subset = leaders[
                (leaders['PredGap'] >= gap) & 
                (leaders['CareerPrize'] >= 20000) &
                (leaders['Odds'] >= odd) & 
                (leaders['Odds'] <= 30)
            ]
            
            if len(subset) < 50: continue
            
            bets = len(subset)
            wins = subset[subset['Position'] == '1'].shape[0]
            strike = (wins / bets) * 100
            profit = subset[subset['Position'] == '1']['Odds'].sum() - bets
            roi = (profit / bets) * 100
            
            print(f"{gap:<8} | {odd:<8} | {bets:<6} | {wins:<7} | {strike:<8.1f} | {roi:<8.1f}")

if __name__ == "__main__":
    run_gap_analysis()
