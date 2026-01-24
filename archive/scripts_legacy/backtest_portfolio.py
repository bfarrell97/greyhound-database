"""
Backtest Portfolio Strategy ("The Sniper") - 2025
Goal: Validate the combined performance of profitable Track/Strategy pairs.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

DB_PATH = 'greyhound_racing.db'

# DERIVED FROM ANALYSIS
PORTFOLIO = {
    'Goulburn': ['Rabbit', 'Steamroller'],     # Huge ROI on both
    'Townsville': ['Rabbit', 'Steamroller'],   # Huge ROI on both
    'Horsham': ['Dominator'],                  # Dominator significantly better
    'Grafton': ['Dominator'],
    'Warrnambool': ['Rabbit'],                 # Rabbit volume >> Dominator
    'Sandown Park': ['Rabbit'],                # Rabbit (10%) > Dominator (3%)
    'Bendigo': ['Steamroller'],                # Steamroller (7%) > Swooper (6%) > Dominator (3%)
    'Northam': ['Dominator'],
    'Sale': ['Swooper'],                       # Swooper (6%) > Dominator (0.5%)
    'Geelong': ['Dominator'],                  # Dominator (6%) > Rabbit/Steamroller (1%)
    'Meadows (MEP)': ['Dominator'],
    'Maitland': ['Dominator'],
    'Taree': ['Steamroller'],                  # Steamroller (1.5%) > Dominator (0.5%) - Marginal
}

def load_data():
    print("Loading Data (2023-2025)...")
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
    WHERE rm.MeetingDate >= '2023-01-01'
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
    print("Engineering Features...")
    
    # 1. Benchmarks
    cols = ['FinishTime', 'Split', 'RunHome']
    for c in cols:
        bench_col = f'TrackDistMedian{c}'
        bench = df.groupby(['TrackName', 'Distance'])[c].median().reset_index()
        bench.columns = ['TrackName', 'Distance', bench_col]
        df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
        df[f'Norm{c}'] = df[c] - df[bench_col]
        
    # 2. Rolling Avgs
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

def backtest(df):
    print("\n" + "="*80)
    print("PORTFOLIO BACKTEST (2025)")
    print("="*80)
    
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Ranks
    df['PredSplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min')
    df['PredRunHomeRank'] = df.groupby('RaceKey')['PredRunHome'].rank(method='min')
    df['PredOverallRank'] = df.groupby('RaceKey')['PredOverall'].rank(method='min')
    
    # Strategy Flags
    df['IsDominator'] = (df['PredSplitRank'] == 1) & (df['PredOverallRank'] == 1)
    df['IsSwooper'] = (df['PredRunHomeRank'] == 1) & (df['PredOverallRank'] == 1) & (df['PredSplitRank'] > 1)
    df['IsRabbit'] = (df['PredSplitRank'] == 1)
    df['IsSteamroller'] = (df['PredRunHomeRank'] == 1)
    
    # Apply Portfolio Logic
    # For each row, check if Track is in Portfolio, AND if corresponding strategy matches
    
    bets = []
    
    # This loop is inefficient but flexible. Vectorized is better.
    # Vectorized approach:
    # 1. Create a 'BetSignal' col default False
    df['BetSignal'] = False
    
    for track, strategies in PORTFOLIO.items():
        track_mask = (df['TrackName'] == track)
        strat_mask = pd.Series(False, index=df.index)
        
        for s in strategies:
            if s == 'Dominator': strat_mask |= df['IsDominator']
            if s == 'Swooper': strat_mask |= df['IsSwooper']
            if s == 'Rabbit': strat_mask |= df['IsRabbit']
            if s == 'Steamroller': strat_mask |= df['IsSteamroller']
            
        final_mask = track_mask & strat_mask & (df['Odds'] >= 2.0) & (df['Odds'] <= 30)
        df.loc[final_mask, 'BetSignal'] = True
        
    portfolio_bets = df[df['BetSignal']].copy()
    
    # Results
    n_bets = len(portfolio_bets)
    wins = portfolio_bets['IsWin'].sum()
    profit = (portfolio_bets[portfolio_bets['IsWin'] == 1]['Odds'] - 1).sum() - (n_bets - wins)
    roi = profit / n_bets * 100 if n_bets > 0 else 0
    
    print(f"Total Bets: {n_bets}")
    print(f"Wins:       {wins}")
    print(f"Strike Rate: {wins/n_bets*100:.1f}%")
    print(f"Profit:     ${profit:.2f}")
    print(f"ROI:        {roi:.1f}%")
    
    # Monthly Breakdown
    portfolio_bets['Month'] = portfolio_bets['MeetingDate'].dt.to_period('M')
    monthly = portfolio_bets.groupby('Month').apply(
        lambda x: pd.Series({
            'Bets': len(x),
            'Profit': (x[x['IsWin']==1]['Odds']-1).sum() - (len(x) - x['IsWin'].sum())
        })
    )
    print("\nMonthly Breakdown:")
    print(monthly)

if __name__ == "__main__":
    df = load_data()
    # If Goulburn/Townsville data is empty for some reason, warn
    if len(df) == 0:
        print("Error: No data found.")
    else:
        df = feature_engineering(df)
        test_df = train_and_predict(df)
        backtest(test_df)
