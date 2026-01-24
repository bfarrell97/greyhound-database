"""
Optimize Dual Dominant Strategy
Test if combining "Dominant Pace Gap" with "Predicted Split Leader" improves Short Course profitability.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'

def run_dual_optimization():
    conn = sqlite3.connect(DB_PATH)
    print("Loading Data (2024-2025)...")
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
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Preprocessing
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
    
    # Feature Eng (Common for both)
    print("Feature Eng...")
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # SPLIT Features
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # PACE Features
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter Valid
    df = df.dropna(subset=['s_Roll5', 'p_Roll5', 'Odds']).copy()
    
    # Predict
    print("Predicting Pace & Split...")
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    with open(PIR_MODEL_PATH, 'rb') as f: pir_model = pickle.load(f)
    
    # X Pace
    X_pace = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormPace'] = pace_model.predict(X_pace)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    # X Split
    X_split = df[['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormSplit'] = pir_model.predict(X_split)
    df['PredSplit'] = df['PredNormSplit'] + df['TrackDistMedianSplit']
    
    # Ranks & Gaps
    print("Calculating Ranks...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Filter Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # Pace Gap Logic (Sort by Pace)
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['RankPace'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['PaceGap'] = df['NextTime'] - df['PredPace']
    
    # Split Rank Logic (Independent Sort)
    df['RankSplit'] = df.groupby('RaceKey')['PredSplit'].rank(method='min', ascending=True)
    
    # Strategy Simulation
    leaders = df[df['RankPace'] == 1].copy()
    
    # Define Baseline: Short Course Dominant (Gap > 0.10, Dist < 450)
    # Looser filters to get volume for comparison
    base_mask = (leaders['Distance'] < 450) & (leaders['PaceGap'] >= 0.10) & (leaders['CareerPrize'] >= 20000)
    
    print("\n" + "="*80)
    print("DUAL MODEL TEST (Dist < 450m, Prize > 20k, Pace Gap > 0.10s)")
    print("="*80)
    print(f"{'Strategy':<25} | {'Min Odds':<8} | {'Bets':<6} | {'Winners':<7} | {'Strike %':<8} | {'ROI %':<8} | {'Profit':<8}")
    print("-" * 100)
    
    odds_levels = [2.00]
    
    with open('results/dual_results.txt', 'w') as f:
        f.write("DEBUG START\n")
        for odd in odds_levels:
            f.write(f"DEBUG: Processing Odds {odd}...\n")
            # 1. Base Strategy (Pace Only)
            base = leaders[base_mask & (leaders['Odds'] >= odd) & (leaders['Odds'] <= 30)]
            f.write(f"DEBUG: Base Count {len(base)}\n")
            
            if len(base) > 0:
                wins = base[base['Position'] == '1'].shape[0]
                roi = ((base[base['Position'] == '1']['Odds'].sum() - len(base)) / len(base)) * 100
                f.write(f"{'Pace Only':<25} | {odd:<8} | {len(base):<6} | {wins:<7} | {(wins/len(base))*100:<8.1f} | {roi:<8.1f} | {base[base['Position'] == '1']['Odds'].sum() - len(base):<8.1f}\n")
            else:
                f.write("Pace Only: No bets found\n")

            # 2. Dual Strategy (Pace + Split Rank 1)
            dual = base[base['RankSplit'] == 1]
            f.write(f"DEBUG: Dual Count {len(dual)}\n")
            
            if len(dual) > 0:
                wins = dual[dual['Position'] == '1'].shape[0]
                roi = ((dual[dual['Position'] == '1']['Odds'].sum() - len(dual)) / len(dual)) * 100
                f.write(f"{'Dual (Pace+Split Ldr)':<25} | {odd:<8} | {len(dual):<6} | {wins:<7} | {(wins/len(dual))*100:<8.1f} | {roi:<8.1f} | {dual[dual['Position'] == '1']['Odds'].sum() - len(dual):<8.1f}\n")
            else:
                f.write("Dual: No bets found\n")
                
            f.write("-" * 100 + "\n")
            
    print("Results written to results/dual_results.txt")


if __name__ == "__main__":
    run_dual_optimization()
