"""
Test Pace Model V2 ROI
Check if the Improved Pace Model (MAE 0.2614) yields better ROI on the Dominant Leader Strategy than V1.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_V2_PATH = 'models/pace_v2_xgb_model.pkl'

def test_v2_roi():
    conn = sqlite3.connect(DB_PATH)
    print("Loading Data...")
    # Need new columns: Weight, Position for WinRate
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
        ge.Weight,
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
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(30.0)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    print("Feature Eng V2...")
    # Benchmarks
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # Pace Features
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Std5']  = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).std().fillna(0))
    
    # Split Features
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    
    # Weight
    df['AvgWeight'] = g['Weight'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['WeightDiff'] = df['Weight'] - df['AvgWeight'].fillna(df['Weight'])
    
    # WinRate
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df['WinRate'] = g['IsWin'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    
    # Other
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter
    df = df.dropna(subset=['p_Roll5', 's_Roll3', 'Odds']).copy()
    
    # Predict V2
    print("Predicting V2...")
    with open(PACE_MODEL_V2_PATH, 'rb') as f: model = pickle.load(f)
    
    features = [
        'p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'p_Std5',
        's_Lag1', 's_Roll3',
        'Weight', 'WeightDiff',
        'WinRate',
        'DaysSince', 'Box', 'Distance'
    ]
    
    df['PredNormPace'] = model.predict(df[features])
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    # Rank & Gap
    print("Calculating Gaps...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Filter Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    # 2. Test Dominant Strategy V2
    # Filters found in V1 tuning: Dist < 450, Gap > 0.15, Prize > 20k, Odds > 2.0
    
    leaders = df[df['Rank'] == 1].copy()
    
    mask = (leaders['Distance'] < 450) & (leaders['Gap'] >= 0.15) & (leaders['CareerPrize'] >= 20000) & (leaders['Odds'] >= 2.0) & (leaders['Odds'] <= 30)
    
    final = leaders[mask]
    
    print("\n" + "="*80)
    print("V2 MODEL DOMINANT STRATEGY RESULTS")
    print("="*80)
    
    if len(final) == 0:
        print("No bets found.")
    else:
        bets = len(final)
        wins = final[final['Position'] == '1'].shape[0]
        strike = (wins / bets) * 100
        profit = final[final['Position'] == '1']['Odds'].sum() - bets
        roi = (profit / bets) * 100
        
        print(f"Bets: {bets}")
        print(f"Wins: {wins}")
        print(f"Strike: {strike:.1f}%")
        print(f"P/L: {profit:.1f}u")
        print(f"ROI: {roi:.1f}%")
        
        # Compare with V1 baseline (approx from previous)
        # V1 had ROI +19.8% on Odds > 2.0 (221 bets)
        print(f"(V1 Baseline: ~220 bets, ~+20% ROI)")

if __name__ == "__main__":
    test_v2_roi()
