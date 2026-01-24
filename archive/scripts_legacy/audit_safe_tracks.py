"""
Audit Safe Tracks (Warrnambool, Meadows, Geelong)
Goal: Verify profit is genuine and not driven by outliers or data errors.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

# The 3 positive tracks from the Safe Backtest
AUDIT_TARGETS = {
    'Warrnambool': ['Dominator', 'Rabbit'],
    'Meadows (MEP)': ['Dominator'],
    'Geelong': ['Dominator']
}

def load_data():
    print("Loading Data (2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
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
      AND t.TrackName IN ('Warrnambool', 'Meadows (MEP)', 'Geelong')
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
    
    return df

def train_and_audit(df):
    # Train 2024, Test 2025
    train_df = df[df['MeetingDate'].dt.year == 2024].copy()
    test_df = df[df['MeetingDate'].dt.year == 2025].copy()
    
    # Drop rows without targets for training
    train_df = train_df.dropna(subset=['NormSplit', 'NormRunHome', 'NormFinishTime'])
    
    print(f"Training Size: {len(train_df)}")
    
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'SireWinRate', 'Box', 'Distance']
    targets = {'Split': 'NormSplit', 'RunHome': 'NormRunHome', 'Overall': 'NormFinishTime'}
    
    for name, target in targets.items():
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', 
            n_estimators=100, learning_rate=0.1, max_depth=3, n_jobs=-1, tree_method='hist'
        )
        model.fit(train_df[features], train_df[target])
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    # Rank and Strategy
    test_df['RaceKey'] = test_df['MeetingDate'].astype(str) + '_' + test_df['TrackName'] + '_' + test_df['RaceID'].astype(str)
    test_df['PredSplitRank'] = test_df.groupby('RaceKey')['PredSplit'].rank(method='min')
    test_df['PredRunHomeRank'] = test_df.groupby('RaceKey')['PredRunHome'].rank(method='min')
    test_df['PredOverallRank'] = test_df.groupby('RaceKey')['PredOverall'].rank(method='min')
    
    test_df['IsDominator'] = (test_df['PredSplitRank'] == 1) & (test_df['PredOverallRank'] == 1)
    test_df['IsRabbit'] = (test_df['PredSplitRank'] == 1)
    
    # Audit Each Track
    for track, strats in AUDIT_TARGETS.items():
        print("\n" + "="*80)
        print(f"AUDIT: {track.upper()}")
        print("="*80)
        
        track_mask = (test_df['TrackName'] == track)
        strat_mask = pd.Series(False, index=test_df.index)
        for s in strats:
            if s == 'Dominator': strat_mask |= test_df['IsDominator']
            if s == 'Rabbit': strat_mask |= test_df['IsRabbit']
            
        bets = test_df[track_mask & strat_mask & (test_df['Odds'] >= 2.0) & (test_df['Odds'] <= 30)].copy()
        
        wins = bets['IsWin'].sum()
        profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum() - (len(bets) - wins)
        roi = profit / len(bets) * 100 if len(bets) > 0 else 0
        
        print(f"Bets: {len(bets)} | Wins: {wins} | Profit: ${profit:.2f} | ROI: {roi:.1f}%")
        
        if len(bets) > 0:
            print("\nTop 10 Biggest Winners:")
            winners = bets[bets['IsWin'] == 1].sort_values('Odds', ascending=False).head(10)
            print(winners[['MeetingDate', 'GreyhoundName', 'Box', 'Odds']].to_string())
            
            # Check for outliers
            outliers = winners[winners['Odds'] > 20]
            if len(outliers) > 0:
                print(f"\n[WARNING] {len(outliers)} winners > $20. Profit may be fragile.")
            else:
                print("\n[OK] No extreme outliers (> $20).")
                
            # Verify Splits on Losers AGAIN for this specific track
            losers = bets[bets['IsWin'] == 0]
            win_splits = bets[bets['IsWin'] == 1]['Split'].notna().mean()
            lose_splits = losers['Split'].notna().mean()
            print(f"\nSplit Data Coverage (Bets Only):")
            print(f"  Winners: {win_splits*100:.1f}%")
            print(f"  Losers:  {lose_splits*100:.1f}%")
            if abs(win_splits - lose_splits) > 0.1:
                print("  [CRITICAL] Survivor Bias DETECTED in bets subset.")
            else:
                print("  [OK] Data appears balanced.")

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        df = feature_engineering(df)
        train_and_audit(df)
    else:
        print("No data.")
