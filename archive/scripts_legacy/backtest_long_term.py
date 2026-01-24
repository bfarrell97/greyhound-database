"""
Long-Term Walk-Forward Validation (2021-2025)
Goal: Validate "The Sniper" Portfolio Strategy across 5 years using Expanding Window.
Data starts 2020. First test year = 2021 (Train on 2020).
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

DB_PATH = 'greyhound_racing.db'

# THE SNIPER PORTFOLIO (Derived from 2025 analysis)
PORTFOLIO = {
    'Goulburn': ['Rabbit', 'Steamroller'],
    'Townsville': ['Rabbit', 'Steamroller'],
    'Horsham': ['Dominator'],
    'Grafton': ['Dominator'],
    'Warrnambool': ['Rabbit'],
    'Sandown Park': ['Rabbit'],
    'Bendigo': ['Steamroller'],
    'Northam': ['Dominator'],
    'Sale': ['Swooper'],
    'Geelong': ['Dominator'],
    'Meadows (MEP)': ['Dominator'],
    'Maitland': ['Dominator'],
    'Taree': ['Steamroller'],
}

def load_data():
    print("Loading Data (2020-2025)...")
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
    WHERE rm.MeetingDate >= '2020-01-01'
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
        
    # 3. Sire Stats (Expanding Window to prevent leakage)
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    df = df.dropna(subset=['Split', 'FinishTime', 'RunHome']).copy()
    return df

def train_and_predict_year(df, test_year):
    print(f"\nProcessing Year: {test_year}")
    
    # Sliding Window: Train on last 2 years only (Speed Optimization)
    # e.g. For 2025, train on 2023-2024.
    start_train_year = test_year - 2
    train_mask = (df['MeetingDate'].dt.year >= start_train_year) & (df['MeetingDate'].dt.year < test_year)
    test_mask = (df['MeetingDate'].dt.year == test_year)
    
    train_df = df[train_mask]
    test_df = df[test_mask].copy()
    
    print(f"  Train Period: {start_train_year}-{test_year-1}")
    print(f"  Train Size:   {len(train_df)}")
    print(f"  Test Size:    {len(test_df)}")
    
    if len(train_df) < 1000 or len(test_df) == 0:
        print("  Not enough data. Skipping.")
        return pd.DataFrame()

    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'SireWinRate', 'Box', 'Distance']
    targets = {'Split': 'NormSplit', 'RunHome': 'NormRunHome', 'Overall': 'NormFinishTime'}
    
    for name, target in targets.items():
        # Tree method 'hist' is much faster for larger datasets
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            n_jobs=-1, 
            verbosity=0,
            tree_method='hist' 
        )
        t_df = train_df.dropna(subset=[target])
        model.fit(t_df[features], t_df[target])
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    return test_df

def run_backtest(df):
    print("\n" + "="*80)
    print("LONG-TERM WALK-FORWARD BACKTEST (2021-2025)")
    print("="*80)
    
    years = [2021, 2022, 2023, 2024, 2025]
    all_bets = []
    
    for year in years:
        year_results = train_and_predict_year(df, year)
        if year_results.empty: continue
        
        # Strategy Logic
        year_results['RaceKey'] = year_results['MeetingDate'].astype(str) + '_' + year_results['TrackName'] + '_' + year_results['RaceID'].astype(str)
        
        # Ranks
        year_results['PredSplitRank'] = year_results.groupby('RaceKey')['PredSplit'].rank(method='min')
        year_results['PredRunHomeRank'] = year_results.groupby('RaceKey')['PredRunHome'].rank(method='min')
        year_results['PredOverallRank'] = year_results.groupby('RaceKey')['PredOverall'].rank(method='min')
        
        # Labels
        year_results['IsDominator'] = (year_results['PredSplitRank'] == 1) & (year_results['PredOverallRank'] == 1)
        year_results['IsSwooper'] = (year_results['PredRunHomeRank'] == 1) & (year_results['PredOverallRank'] == 1) & (year_results['PredSplitRank'] > 1)
        year_results['IsRabbit'] = (year_results['PredSplitRank'] == 1)
        year_results['IsSteamroller'] = (year_results['PredRunHomeRank'] == 1)
        
        # Portfolio Selection
        year_results['BetSignal'] = False
        
        for track, strategies in PORTFOLIO.items():
            track_mask = (year_results['TrackName'] == track)
            strat_mask = pd.Series(False, index=year_results.index)
            
            for s in strategies:
                if s == 'Dominator': strat_mask |= year_results['IsDominator']
                if s == 'Swooper': strat_mask |= year_results['IsSwooper']
                if s == 'Rabbit': strat_mask |= year_results['IsRabbit']
                if s == 'Steamroller': strat_mask |= year_results['IsSteamroller']
            
            final_mask = track_mask & strat_mask & (year_results['Odds'] >= 2.0) & (year_results['Odds'] <= 30)
            year_results.loc[final_mask, 'BetSignal'] = True
            
        bets = year_results[year_results['BetSignal']].copy()
        
        # Year Stats
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum() - (n_bets - wins)
        roi = profit / n_bets * 100 if n_bets > 0 else 0
        
        print(f"YEAR: {year} | Bets: {n_bets:<6} | Wins: {wins:<6} | Strike: {wins/n_bets*100:<5.1f}% | Profit: {profit:<8.2f} | ROI: {roi:<6.1f}%")
        
        all_bets.append(bets)
        
    # Total
    full_df = pd.concat(all_bets)
    n_bets = len(full_df)
    wins = full_df['IsWin'].sum()
    profit = (full_df[full_df['IsWin'] == 1]['Odds'] - 1).sum() - (n_bets - wins)
    roi = profit / n_bets * 100 if n_bets > 0 else 0
    
    print("-" * 80)
    print(f"TOTAL (2021-2025) | Bets: {n_bets:<6} | Wins: {wins:<6} | Strike: {wins/n_bets*100:<5.1f}% | Profit: {profit:<8.2f} | ROI: {roi:<6.1f}%")
    
    print("\n" + "="*80)
    print("TRACK BREAKDOWN (2021-2025)")
    print("="*80)
    print(f"{'Track':<20} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'Profit':<10} {'ROI%':<10}")
    print("-" * 80)
    
    track_stats = full_df.groupby('TrackName').agg(
        Bets=('IsWin', 'count'),
        Wins=('IsWin', 'sum'),
        TotalWinningOdds=('Odds', lambda x: x[full_df.loc[x.index, 'IsWin'] == 1].sum())
    )
    
    # Calculate Profit manually from Aggregates to avoid convoluted lambda
    # Profit = (Sum of Winning Odds) - (Total Bets)
    # Note: The lambda above gets sum of odds for winners. 
    # But it's safer to just iterate or do vector calc on the result.
    
    # Let's do a safer vector approach:
    # 1. Calculate Profit per bet in original df
    full_df['Profit'] = np.where(full_df['IsWin'] == 1, full_df['Odds'] - 1, -1)
    
    track_stats = full_df.groupby('TrackName').agg(
        Bets=('IsWin', 'count'),
        Wins=('IsWin', 'sum'),
        Profit=('Profit', 'sum')
    )
    track_stats = track_stats.sort_values('Profit', ascending=False)
    
    for track, row in track_stats.iterrows():
        bets = int(row['Bets'])
        wins = int(row['Wins'])
        profit = row['Profit']
        roi = profit / bets * 100 if bets > 0 else 0
        strike = wins / bets * 100 if bets > 0 else 0
        
        print(f"{track:<20} {bets:<8} {wins:<8} {strike:<10.1f} {profit:<10.1f} {roi:<10.1f}")

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        df = feature_engineering(df)
        run_backtest(df)
    else:
        print("No data.")
