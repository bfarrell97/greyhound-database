import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading 2025 Data for equivalence check...")
    with open('tier1_tracks.txt', 'r') as f:
        safe_tracks = [line.strip() for line in f if line.strip()]

    conn = sqlite3.connect(DB_PATH)
    placeholders = ',' .join('?' for _ in safe_tracks)
    # Load enough data to train (2022-2024ish) and test 2025
    query = f"""
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
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
      AND t.TrackName IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=safe_tracks)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # Simple Benchmark Feature
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    return df.dropna(subset=['DogNormTimeAvg'])

def run_simulation(df):
    years = [2021, 2022, 2023, 2024, 2025]
    
    print(f"\nRunning 5-Year Walk-Forward Validation ({years[0]}-{years[-1]})")
    print(f"{'Year':<6} {'Bets':<6} {'Strike':<8} {'Profit':<10} {'ROI%':<6}")
    print("-" * 50)
    
    all_results = []
    
    # We need to accumulate predictions across all years
    all_test_preds = []

    for test_year in years:
        # Separate Train/Test
        train_mask = df['MeetingDate'].dt.year < test_year
        test_mask = df['MeetingDate'].dt.year == test_year
        
        train = df[train_mask]
        test = df[test_mask].copy()
        
        if len(test) == 0:
            print(f"{test_year}: No Data")
            continue
            
        if len(train) < 1000:
             print(f"{test_year}: Insufficient training data ({len(train)})")
             continue

        # Train Model
        features = ['DogNormTimeAvg', 'Box', 'Distance']
        model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, tree_method='hist')
        model.fit(train[features], train['NormTime'])
        
        # Predict
        test['PredOverall'] = model.predict(test[features])
        
        # Calculate Ranks (needed for Strategy)
        test['PredRank'] = test.groupby('RaceID')['PredOverall'].rank(method='min')
        
        # Collect for aggregate analysis
        all_test_preds.append(test)
        
        # Evaluate stats for THIS year (GUI Strategy Only: Rank 1 < 2.25)
        strat_a = test[
            (test['PredRank'] == 1) &
            (test['Odds'] < 2.25) &
            (test['Odds'] >= 1.50)
        ].copy()
        
        if len(strat_a) == 0:
             print(f"{test_year:<6} 0      0.0%     $0         0.0%")
             continue
             
        wins = (strat_a['IsWin'] == 0).sum()
        stake = 100
        comm = 0.05
        gross = wins * stake * (1-comm)
        strat_a['Liability'] = (strat_a['Odds'] - 1) * stake
        losses = strat_a[strat_a['IsWin'] == 1]['Liability'].sum()
        net = gross - losses
        risk = strat_a['Liability'].sum()
        roi = net/risk*100 if risk > 0 else 0
        strike = wins/len(strat_a)*100
        
        print(f"{test_year:<6} {len(strat_a):<6} {strike:<8.1f} ${net:<9.0f} {roi:<6.1f}%")
        
        all_results.append({
            'Year': test_year,
            'Bets': len(strat_a),
            'Profit': net,
            'Risk': risk
        })

    # Total Stats
    total_profit = sum(r['Profit'] for r in all_results)
    total_risk = sum(r['Risk'] for r in all_results)
    total_roi = total_profit / total_risk * 100 if total_risk > 0 else 0
    
    print("-" * 50)
    print("-" * 50)
    print(f"TOTAL  {sum(r['Bets'] for r in all_results):<6} {'-':<8} ${total_profit:<9.0f} {total_roi:<6.1f}%")

    # Stack all bets for stress testing
    if all_test_preds:
         # We need to reconstruct the strat_a filter on the full set
         full_test = pd.concat(all_test_preds)
         final_trades = full_test[
            (full_test['PredRank'] == 1) &
            (full_test['Odds'] < 2.25) &
            (full_test['Odds'] >= 1.50)
         ].copy()
         
         final_trades.to_csv("all_trades.csv", index=False)
         print(f"Saved {len(final_trades)} trades to all_trades.csv for Stress Testing.")

if __name__ == "__main__":
    df = load_data()
    run_simulation(df)
