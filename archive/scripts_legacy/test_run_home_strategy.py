"""
Test Run Home (Swooper) Strategy (2025)
Goal: Test if backing strong finishers (Swoopers) offers better ROI than leaders.
Hypothesis: Market overbets early speed, leaving value on late speed.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading 2024-2025 Data...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.SireID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Split,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
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
    
    # Run Home
    df['RunHome'] = df['FinishTime'] - df['Split']
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Run Home Features...")
    
    # 1. Benchmarks
    rh_bench = df.groupby(['TrackName', 'Distance'])['RunHome'].median().reset_index()
    rh_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianRunHome']
    df = df.merge(rh_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormRunHome'] = df['RunHome'] - df['TrackDistMedianRunHome']
    
    # 2. Rolling Avg (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['DogRunHomeAvg'] = g['NormRunHome'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # Also calc Pace for comparison/filtering?
    # Let's keep it pure Swooper for now.
    
    # 3. Sire Stats (Expanding)
    print("  Calculating Sire Stats...")
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Filter only 2025 for testing
    test_df = df[df['MeetingDate'].dt.year == 2025].copy()
    return test_df

def test_strategies(df):
    print("\n" + "="*80)
    print("SWOOPER STRATEGY PERFORMANCE (2025)")
    print("="*80)
    
    # Define RaceKey
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Filter Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6].copy()
    
    # Rank RunHome (Lowest is Strongest/Fastest Run Home? 
    # Yes, RunHome = Time. Lower is faster.)
    df['SwoopRank'] = df.groupby('RaceKey')['DogRunHomeAvg'].rank(method='min')
    
    # Sire Rank
    # Highest WinRate is Rank 1
    df['SireRank'] = df.groupby('RaceKey')['SireWinRate'].rank(ascending=False, method='min')
    
    strategies = {
        'Swooper (Rank 1 RH)': (df['SwoopRank'] == 1),
        'Elite Swooper (Rank 1 RH + Top 3 Sire)': (df['SwoopRank'] == 1) & (df['SireRank'] <= 3),
        'Rail Swooper (Rank 1 RH + Box 1/2)': (df['SwoopRank'] == 1) & (df['Box'].isin([1, 2])),
        'Wide Swooper (Rank 1 RH + Box 7/8)': (df['SwoopRank'] == 1) & (df['Box'].isin([7, 8])),
        'Value Swooper (Rank 1 RH + Odds > 4.0)': (df['SwoopRank'] == 1) & (df['Odds'] >= 4.0),
    }
    
    print(f"{'Strategy':<40} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'Profit':<10} {'ROI%':<10}")
    print("-" * 88)
    
    for name, mask in strategies.items():
        # Apply Odds Filter (Standard safety)
        # Maybe relax upper bound for swoopers?
        bet_mask = mask & (df['Odds'] >= 2.0) & (df['Odds'] <= 50)
        bets = df[bet_mask].copy()
        
        if len(bets) == 0:
            print(f"{name:<40} 0 bets")
            continue
            
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        strike = wins / n_bets * 100
        
        winner_profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum()
        loser_loss = (n_bets - wins)
        profit = winner_profit - loser_loss
        roi = profit / n_bets * 100
        
        print(f"{name:<40} {n_bets:<8} {wins:<8} {strike:<10.1f} {profit:<10.1f} {roi:<10.1f}")

if __name__ == "__main__":
    df = load_data()
    test_df = feature_engineering(df)
    test_strategies(test_df)
