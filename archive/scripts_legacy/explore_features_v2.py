"""
Exploratory Feature Analysis 2.0 (New Angles)
Goal: Identify predictive features beyond early speed (Run Home, Sire, Weight).
Data: 2024-2025
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
        ge.Weight,
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
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    # Run Home (Finish - Split)
    # Note: If Split is null, RunHome is null
    df['RunHome'] = df['FinishTime'] - df['Split']
    
    return df

def feature_engineering(df):
    print("Engineering Features (Set 2.0)...")
    
    # 1. Run Home Benchmark
    # We want to know if a dog has a 'Strong' Run Home relative to the track/dist avg
    rh_bench = df.groupby(['TrackName', 'Distance'])['RunHome'].median().reset_index()
    rh_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianRunHome']
    df = df.merge(rh_bench, on=['TrackName', 'Distance'], how='left')
    
    # NormRunHome: Negative = Faster than avg (Stronger run home)
    df['NormRunHome'] = df['RunHome'] - df['TrackDistMedianRunHome']
    
    # Rolling RunHome Avg (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['DogRunHomeAvg'] = g['NormRunHome'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    
    # 2. Sire Stats
    # Global average for 2024-2025 (Exploratory look)
    # Expanding would be better for strictly valid test, but for signal check global is usually fine proxy
    # Let's use expanding to be safe and rigorous.
    print("  Calculating Sire Stats (Expanding)...")
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    # Filter Sires with few runners?
    sire_counts = df.groupby('SireID').size()
    valid_sires = sire_counts[sire_counts > 50].index
    df.loc[~df['SireID'].isin(valid_sires), 'SireWinRate'] = np.nan
    
    # 3. Weight
    # Raw weight is interesting.
    # Also Weight vs Track Avg?
    # Let's stick to simple Weight first.
    
    return df

def analyze_correlations(df):
    output = []
    output.append("\n" + "="*80)
    output.append("FEATURE ANALYSIS 2.0 (Run Home, Sire, Weight)")
    output.append("="*80)
    
    features = {
        'DogRunHomeAvg': 'DogRunHomeAvg', # Lower = Stronger Finish
        'SireWinRate': 'SireWinRate', # Higher = Better Genetics
        'Weight': 'Weight', # ?
    }
    
    output.append(f"{'Feature':<40} {'Correlation (IC)':<20} {'Direction'}")
    output.append("-" * 75)
    
    for label, col in features.items():
        valid = df.dropna(subset=[col, 'IsWin'])
        if len(valid) < 1000:
            output.append(f"{label:<40} Not enough data")
            continue
            
        corr = valid[col].corr(valid['IsWin'])
        direction = "Positive (Higher=Win)" if corr > 0 else "Negative (Lower=Win)"
        output.append(f"{label:<40} {corr:<20.4f} {direction}")
        
    output.append("\n" + "="*80)
    output.append("WIN RATE BY DECILE (1 = Lowest Value, 5 = Highest Value)")
    output.append("="*80)
    
    for label, col in features.items():
        valid = df.dropna(subset=[col, 'IsWin']).copy()
        
        try:
            valid['Decile'] = pd.qcut(valid[col], 5, labels=['1 (Low)', '2', '3', '4', '5 (High)'])
            stats = valid.groupby('Decile')['IsWin'].mean()
            
            output.append(f"\n{label} ({col}):")
            for cat, val in stats.items():
                 output.append(f"  {cat}: {val*100:.1f}% Win Rate")
        except Exception as e:
            output.append(f"Error binning {label}: {e}")

    with open('exploration_v2_results.txt', 'w') as f:
        f.write('\n'.join(output))
    print("Results saved to exploration_v2_results.txt")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    analyze_correlations(df)
