"""
Test Simple Strategies (2025)
Goal: Check if simple Ranking logic (Pace/Split Leader) performs better than complex models.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading 2025 Data...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Split,
        ge.Position,
        ge.StartingPrice,
        ge.PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2025-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Features (Naive Rolling)...")
    
    # Benchmarks
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    
    # Rolling 3
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    df['DogPaceAvg'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['DogSplitAvg'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    return df

def test_strategies(df):
    print("\n" + "="*80)
    print("SIMPLE STRATEGY PERFORMANCE (2025)")
    print("="*80)
    
    # Define RaceKey
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Filter Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6].copy()
    
    # Ranks (Lowest Avg is Best)
    df['PaceRank'] = df.groupby('RaceKey')['DogPaceAvg'].rank(method='min')
    df['SplitRank'] = df.groupby('RaceKey')['DogSplitAvg'].rank(method='min')
    
    strategies = {
        'Pace Leader (Rank 1)': (df['PaceRank'] == 1),
        'Split Leader (Rank 1)': (df['SplitRank'] == 1),
        'Dual Leader (Pace+Split 1)': (df['PaceRank'] == 1) & (df['SplitRank'] == 1),
        'Dual Leader + Box 1': (df['PaceRank'] == 1) & (df['SplitRank'] == 1) & (df['Box'] == 1),
        'Dual Leader + Dist<400': (df['PaceRank'] == 1) & (df['SplitRank'] == 1) & (df['Distance'] < 400),
    }
    
    print(f"{'Strategy':<30} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'Profit':<10} {'ROI%':<10}")
    print("-" * 75)
    
    for name, mask in strategies.items():
        # Apply Odds Filter (Standard)
        bet_mask = mask & (df['Odds'] >= 1.5) & (df['Odds'] <= 30)
        bets = df[bet_mask].copy()
        
        if len(bets) == 0:
            print(f"{name:<30} 0 bets")
            continue
            
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        strike = wins / n_bets * 100
        
        profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum() - (n_bets - wins)
        roi = profit / n_bets * 100
        
        print(f"{name:<30} {n_bets:<8} {wins:<8} {strike:<10.1f} {profit:<10.1f} {roi:<10.1f}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    test_strategies(df)
