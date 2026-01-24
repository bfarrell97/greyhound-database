"""
Backtest SAFE Portfolio (Tier 1 Tracks Only)
Goal: Validate strategy performance REMOVING Goulburn/Townsville (Survivor Bias).
Only use tracks confirmed to have >90% Data Completeness.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

# SAFE PORTFOLIO (Derived from Analysis INTERSECT Tier 1 Quality)
# Goulburn, Townsville, Northam REMOVED.
PORTFOLIO = {
    'Horsham': ['Dominator'],                  # +25% in initial scan
    'Grafton': ['Dominator'],                  # +23%
    'Warrnambool': ['Dominator', 'Rabbit'],    # +13%, +11%
    'Sandown Park': ['Rabbit', 'Dominator'],   # +10%, +3%
    'Bendigo': ['Steamroller', 'Swooper'],     # +7.9%, +6.3%
    'Sale': ['Swooper', 'Dominator'],          # +6.8%, +0.5%
    'Geelong': ['Dominator'],                  # +6.2%
    'Meadows (MEP)': ['Dominator'],            # +3.7%
    'Maitland': ['Dominator'],                 # +1.8%
    'Taree': ['Steamroller'],                  # +1.5%
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
    
    # Filter for SAFE tracks only to save memory/speed
    safe_tracks = list(PORTFOLIO.keys())
    df = df[df['TrackName'].isin(safe_tracks)].copy()
    
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
    
    # Train on 2023-2024, Test on 2025
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
            objective='reg:absoluteerror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            n_jobs=-1,
            tree_method='hist'
        )
        t_df = train_df.dropna(subset=[target])
        model.fit(t_df[features], t_df[target])
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    test_df.to_csv('safe_predictions.csv', index=False)
    print("Saved predictions to safe_predictions.csv")
    return test_df

def backtest(df):
    print("\n" + "="*80)
    print("SAFE PORTFOLIO BACKTEST (2025 Test Year)")
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
    
    # Track Breakdown
    print("\nTRACK BREAKDOWN:")
    track_stats = portfolio_bets.groupby('TrackName').agg(
        Bets=('IsWin', 'count'),
        Profit=('Odds', lambda x: (x[portfolio_bets.loc[x.index, 'IsWin']==1]-1).sum() - (len(x)-portfolio_bets.loc[x.index, 'IsWin'].sum()))
    ).sort_values('Profit', ascending=False)
    print(track_stats)

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        df = feature_engineering(df)
        test_df = train_and_predict(df)
        backtest(test_df)
    else:
        print("No data.")
