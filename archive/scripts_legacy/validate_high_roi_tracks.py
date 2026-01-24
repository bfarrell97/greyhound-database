"""
Validate High ROI Tracks (Goulburn, Townsville)
Goal: Inspect specific winners to confirm if ROI is real or due to data errors (e.g. bad Price).
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading Data (2023-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.SireID,
        g.GreyhoundName,
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
    WHERE rm.MeetingDate >= '2023-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND t.TrackName IN ('Goulburn', 'Townsville')
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
    print("Engineering Features...")
    
    # 1. Benchmarks (Global)
    cols = ['FinishTime', 'Split', 'RunHome']
    for c in cols:
        bench_col = f'TrackDistMedian{c}'
        bench = df.groupby(['TrackName', 'Distance'])[c].median().reset_index()
        bench.columns = ['TrackName', 'Distance', bench_col]
        df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
        df[f'Norm{c}'] = df[c] - df[bench_col]
        
    # 2. Rolling Avgs (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    for c in ['NormFinishTime', 'NormSplit', 'NormRunHome']:
        df[f'Dog{c}Avg'] = g[c].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
    # 3. Sire Stats
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    df = df.dropna(subset=['Split', 'FinishTime', 'RunHome']).copy()
    return df

def train_and_predict(df):
    
    train_mask = (df['MeetingDate'].dt.year >= 2023) & (df['MeetingDate'].dt.year <= 2024)
    test_mask = (df['MeetingDate'].dt.year == 2025)
    
    train_df = df[train_mask]
    test_df = df[test_mask].copy()
    
    print(f"Training Size: {len(train_df)}")
    
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'SireWinRate', 'Box', 'Distance']
    targets = {'Split': 'NormSplit', 'RunHome': 'NormRunHome', 'Overall': 'NormFinishTime'}
    
    for name, target in targets.items():
        print(f"  Training {name} Model...")
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=100, learning_rate=0.1, max_depth=3, n_jobs=-1
        )
        t_df = train_df.dropna(subset=[target])
        model.fit(t_df[features], t_df[target])
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    return test_df

def inspect_tracks(df):
    print("\n" + "="*80)
    print("VALIDATION INSPECTION (Goulburn, Townsville 2025)")
    print("="*80)
    
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Ranks
    df['PredSplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min')
    df['PredOverallRank'] = df.groupby('RaceKey')['PredOverall'].rank(method='min')
    
    # Define Strategy: Dominator
    df['IsDominator'] = (df['PredSplitRank'] == 1) & (df['PredOverallRank'] == 1)
    
    # Group by Track
    tracks = ['Goulburn', 'Townsville']
    
    for track in tracks:
        print(f"\nTRACK: {track}")
        print("-" * 60)
        track_df = df[df['TrackName'] == track]
        
        # Analyze Dominator
        # Filter for the bets we would have made
        bets = track_df[track_df['IsDominator'] & (track_df['Odds'] >= 2.0) & (track_df['Odds'] <= 30)].copy()
        
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum() - (len(bets) - wins)
        roi = profit / len(bets) * 100 if len(bets) > 0 else 0
        
        print(f"Bets: {n_bets}, Wins: {wins}, Profit: ${profit:.2f}, ROI: {roi:.1f}%")
        
        # SHOW TOP WINNERS (Suspect ones)
        print("\n  Top 10 Contributors to Profit (High Odds Winners):")
        winners = bets[bets['IsWin'] == 1].sort_values('Odds', ascending=False).head(10)
        print(winners[['MeetingDate', 'RaceID', 'GreyhoundName', 'Box', 'Distance', 'Odds', 'FinishTime', 'Split', 'PredSplitRank', 'PredOverallRank']].to_string())
        
        # SHOW SUSPICIOUS LOSERS (e.g. extremely short odds that lost? Or maybe we just care about winners being fake)
        # Actually, let's look for "Impossible" odds.
        print("\n  Extreme Odds Check (All Bets > $50):")
        extreme = bets[bets['Odds'] > 50]
        if len(extreme) > 0:
            print(extreme[['MeetingDate', 'RaceID', 'GreyhoundName', 'Odds', 'IsWin']])
        else:
            print("  None")

if __name__ == "__main__":
    df = load_data()
    # If Goulburn/Townsville data is empty for some reason, warn
    if len(df) == 0:
        print("Error: No data found for selected tracks.")
    else:
        df = feature_engineering(df)
        test_df = train_and_predict(df)
        inspect_tracks(test_df)
