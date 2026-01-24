"""
Exploratory Feature Analysis (2024-2025)
Goal: Identify predictive features for a new ML model.
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = 'greyhound_racing.db'

def load_recent_data():
    print("Loading 2024-2025 Data...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        ge.TrainerID,
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
    
    # Win Label
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    # Parse Odds
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Features...")
    
    # 1. TRACK/DIST BENCHMARKS (Speed Maps)
    # Global median for 2024-2025
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    
    # 2. ROLLING HISTORY (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # Pace Ratings using 3-race rolling avg
    df['DogPaceAvg'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['DogSplitAvg'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    # 3. TRAINER STATS
    # We want Trainer success rate.
    # Note: This is an optimistic look (using 2024-2025 global stats) for exploration.
    # In a real model, we'd use expanding window to avoid leakage.
    # For "What works?", global correlation is a fine first proxy, but expanding is safer.
    print("  Calculating Trainer Stats (Expanding)...")
    df = df.sort_values(['TrainerID', 'MeetingDate'])
    df['TrainerWinRate'] = df.groupby('TrainerID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # 4. MARKET
    df['ImpliedProb'] = 1 / df['Odds']
    
    return df

def analyze_correlations(df):
    output = []
    output.append("\n" + "="*80)
    output.append("FEATURE CORRELATIONS WITH WINNING (2024-2025)")
    output.append("="*80)
    
    # Filter valid rows for analysis
    features = {
        'DogPaceAvg': 'DogPaceAvg', # Lower is better? We'll check corr
        'DogSplitAvg': 'DogSplitAvg', # Lower is better
        'TrainerWinRate': 'TrainerWinRate', # Higher is better
        'ImpliedProb': 'ImpliedProb', # Higher is better
        'Box': 'Box'
    }
    
    output.append(f"{'Feature':<40} {'Correlation (IC)':<20} {'Direction'}")
    output.append("-" * 75)
    
    for label, col in features.items():
        valid = df.dropna(subset=[col, 'IsWin'])
        if len(valid) < 1000:
            continue
            
        corr = valid[col].corr(valid['IsWin'])
        direction = "Positive (Higher=Win)" if corr > 0 else "Negative (Lower=Win)"
        output.append(f"{label:<40} {corr:<20.4f} {direction}")
        
    output.append("\n" + "="*80)
    output.append("WIN RATE BY DECILE (1 = Lowest Value, 5 = Highest Value)")
    output.append("="*80)
    
    for label, col in features.items():
        if col == 'Box': continue
        
        valid = df.dropna(subset=[col, 'IsWin']).copy()
        try:
            # qcut sorts Low to High.
            # Label 1 = Lowest Value (e.g. Fastest Time if negative normalized? No, usually Pace is seconds.)
            # Wait, NormTime = Time - Benchmark. Negative is Faster.
            # So Lowest Value (Decile 1) = Fastest.
            # For TrainerWinRate, Lowest Value (Decile 1) = Worst Trainer.
            valid['Decile'] = pd.qcut(valid[col], 5, labels=['1 (Low)', '2', '3', '4', '5 (High)'])
            stats = valid.groupby('Decile')['IsWin'].mean()
            output.append(f"\n{label} ({col}):")
            for cat, val in stats.items():
                 output.append(f"  {cat}: {val*100:.1f}% Win Rate")
        except Exception as e:
            output.append(f"Error binning {label}: {e}")

    with open('exploration_results.txt', 'w') as f:
        f.write('\n'.join(output))
    print("Results saved to exploration_results.txt")

if __name__ == "__main__":
    df = load_recent_data()
    df = feature_engineering(df)
    analyze_correlations(df)
