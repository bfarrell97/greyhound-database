"""
Validate Pace Dominator Strategy (Box 1 Specialist)
Method: Walk-Forward Validation (2023-2025) on Safe Tracks.
Goal: Verify stability of the +45% ROI edge.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

# THE STRATEGY RULES
BOX_FILTER = [1]
ODDS_MIN = 3.0
ODDS_MAX = 8.0
MARGIN_THRESH = 0.10 # Seconds faster than 2nd place

def load_data():
    print("Loading Data (2021-2025)...")
    # Load safe tracks
    with open('tier1_tracks.txt', 'r') as f:
        safe_tracks = [line.strip() for line in f if line.strip()]
        
    conn = sqlite3.connect(DB_PATH)
    placeholders = ',' .join('?' for _ in safe_tracks)
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
        ge.Split,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND t.TrackName IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=safe_tracks)
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
    df = df.sort_values(['MeetingDate', 'RaceID'])
    
    # 1. Benchmarks
    cols = ['FinishTime', 'Split', 'RunHome']
    for c in cols:
        bench_col = f'TrackDistMedian{c}'
        bench = df.groupby(['TrackName', 'Distance'])[c].median().reset_index()
        bench.columns = ['TrackName', 'Distance', bench_col]
        df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
        df[f'Norm{c}'] = df[c] - df[bench_col]
    
    # 2. Rolling Dog Stats
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    for c in ['NormFinishTime', 'NormSplit', 'NormRunHome']:
        df[f'Dog{c}Avg'] = g[c].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
    df = df.dropna(subset=['Split', 'FinishTime', 'RunHome']).copy()
    return df

def run_validation(df):
    results = []
    
    # Validation Years
    years = [2023, 2024, 2025]
    
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'Box', 'Distance']
    target = 'NormFinishTime'
    
    for year in years:
        print(f"\nProcessing Year: {year}")
        print("-" * 50)
        
        # Walk-Forward Window: Train on previous 2 years
        train_start = year - 2
        train_end = year - 1
        
        train_mask = (df['MeetingDate'].dt.year >= train_start) & (df['MeetingDate'].dt.year <= train_end)
        test_mask = (df['MeetingDate'].dt.year == year)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        t_subset = train_df.dropna(subset=[target, *features])
        
        print(f"  Train: {train_start}-{train_end} (n={len(t_subset)})")
        print(f"  Test:  {year} (n={len(test_df)})")
        
        # Train Model (Pace - Overall Time)
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=100, learning_rate=0.1, max_depth=3, n_jobs=-1, tree_method='hist'
        )
        model.fit(t_subset[features], t_subset[target])
        test_df['PredOverall'] = model.predict(test_df[features])
        
        # Logic: Rank per RaceID
        test_df['PredRank'] = test_df.groupby('RaceID')['PredOverall'].rank(method='min')
        
        # ---------------------------------------------------------
        # OPTIMIZED MARGIN CALCULATION
        # ---------------------------------------------------------
        # Identify Rank 1s
        rank1s = test_df[test_df['PredRank'] == 1].copy()
        
        # Identify Rank 2s
        rank2s = test_df[test_df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
        rank2s.columns = ['RaceID', 'Time2nd']
        
        # Merge
        result = rank1s.merge(rank2s, on='RaceID', how='left')
        
        # Calculate Margin
        result['Margin'] = result['Time2nd'] - result['PredOverall']
        
        # Populate test_df (actually we only care about rank1s now for betting)
        # So we can just use `result` as our betting candidates base
        test_df = result
        
        # DEBUG BLITZ
        print(f"  DEBUG: Races Analyzed={test_df['RaceID'].nunique()}")
        print(f"  DEBUG: Candidates (Rank1)={len(test_df)}")
        print(f"  DEBUG: Margins > 0.1s={(test_df['Margin'] > 0.1).sum()}")
        
        # ---------------------------------------------------------
        # LAY STRATEGY VALIDATION
        # ---------------------------------------------------------
        # Rules: Lay "Dominators" (Rank 1, Mg > X) at Low Odds (< Y)
        
        configs = [
            {'Margin': 0.1, 'MaxOdds': 4.0},
            {'Margin': 0.1, 'MaxOdds': 6.0},
            {'Margin': 0.2, 'MaxOdds': 4.0}
        ]
        
        STAKE = 100
        COMM = 0.05
        
        for cfg in configs:
            marg = cfg['Margin']
            max_odds = cfg['MaxOdds']
            
            # Filter
            # Note: We used to filter Box 1, now let's try All Boxes for Laying?
            # Or stick to Box 1 False Favorites? 
            # Let's try All Boxes since model confidence is the key signal.
            
            lays = test_df[
                (test_df['PredRank'] == 1) &
                (test_df['Margin'] > marg) &
                (test_df['Odds'] >= 1.50) &
                (test_df['Odds'] <= max_odds)
            ].copy()
            
            if len(lays) == 0: continue
            
            n_bets = len(lays)
            
            # Lay Logic
            # Win if Dog Loses (IsWin == 0)
            lay_wins = (lays['IsWin'] == 0).sum()
            
            # Profit = Wins * Stake * 0.95
            gross_profit = lay_wins * STAKE * (1 - COMM)
            
            # Loss = (Odds - 1) * Stake where IsWin == 1
            lays['Liability'] = (lays['Odds'] - 1) * STAKE
            loss_liability = lays[lays['IsWin'] == 1]['Liability'].sum()
            
            net_profit = gross_profit - loss_liability
            total_risk = lays['Liability'].sum() # Total Liability exposed
            
            roi = net_profit / total_risk * 100 if total_risk > 0 else 0
            
            results.append({
                'Year': year,
                'Config': f"Lay Mg>{marg} <${max_odds}",
                'Bets': n_bets,
                'Profit': net_profit,
                'ROI': roi
            })

    # Summary
    print("\n" + "="*80)
    print("LAY STRATEGY VALIDATION SUMMARY (2023-2025)")
    print("="*80)
    res_df = pd.DataFrame(results)
    
    pivot = res_df.pivot(index='Config', columns='Year', values=['ROI', 'Bets', 'Profit'])
    print(pivot.to_string())
    
    print("\nTotal Aggregated:")
    agg = res_df.groupby('Config').agg({'Bets': 'sum', 'Profit': 'sum', 'ROI': 'mean'}).reset_index()
    print(agg.to_string(index=False))

if __name__ == "__main__":
    df = load_data()
    if len(df) > 0:
        df = feature_engineering(df)
        run_validation(df)
    else:
        print("No data.")
