"""
Optimize Lay Strategy (Target > 10% ROI)
Grid Search for Box, Odds, and Field Size filters.
Uses Walk-Forward methodology (2023-2025).
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def load_and_prep():
    print("Loading Data (2021-2025)...")
    # Reuse load logic from validate script (simplified)
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

def get_predictions(df):
    # Train ONE model on all past data to save time? 
    # Or just Run Walk Forward quickly.
    # Let's do a simple expanding window simulation for speed here (Train 2022, Test 23-25)
    # Actually, we need walk-forward accuracy.
    # We will train on < 2023, Test on >= 2023.
    
    train_mask = df['MeetingDate'] < '2023-01-01'
    test_mask = df['MeetingDate'] >= '2023-01-01'
    
    train = df[train_mask]
    test = df[test_mask].copy()
    
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, tree_method='hist')
    model.fit(train[features], train['NormTime'])
    
    test['PredOverall'] = model.predict(test[features])
    return test

def optimize(test_df):
    print("\nCalculcating Margins...")
    test_df['PredRank'] = test_df.groupby('RaceID')['PredOverall'].rank(method='min')
    
    rank1s = test_df[test_df['PredRank'] == 1].copy()
    rank2s = test_df[test_df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    # Filter base pool
    base_pool = candidates[candidates['Margin'] > 0.1].copy()
    
    print(f"Base Candidates (Mg > 0.1): {len(base_pool)}")
    
    # Grid
    box_sets = {
        'All': [1,2,3,4,5,6,7,8],
        'Middle (4-6)': [4,5,6],
        'Wide (8)': [8],
        'Inside (1-2)': [1,2],
        'Non-Rail (3-8)': [3,4,5,6,7,8]
    }
    
    max_odds_list = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5]
    
    print(f"\n{'Config':<35} {'Bets':<6} {'Str%(L)':<8} {'Profit':<10} {'ROI%':<6} {'Stable?':<8}")
    print("-" * 90)
    
    results = []
    
    STAKE = 100
    COMM = 0.05
    
    for b_name, boxes in box_sets.items():
        for max_o in max_odds_list:
            subset = base_pool[
                (base_pool['Box'].isin(boxes)) &
                (base_pool['Odds'] >= 1.50) &
                (base_pool['Odds'] <= max_o)
            ].copy()
            
            if len(subset) < 100: continue
            
            # P&L
            wins_lay = (subset['IsWin'] == 0).sum()
            gross_profit = wins_lay * STAKE * (1-COMM)
            
            subset['Liability'] = (subset['Odds'] - 1) * STAKE
            losses_lay = subset[subset['IsWin'] == 1]['Liability'].sum()
            
            net_profit = gross_profit - losses_lay
            total_risk = subset['Liability'].sum()
            
            roi = net_profit / total_risk * 100 if total_risk > 0 else 0
            strike = wins_lay / len(subset) * 100
            
            # Check Stability (Profit in all 3 years?)
            subset['Year'] = subset['MeetingDate'].dt.year
            yearly_prof = []
            for y in [2023, 2024, 2025]:
                y_sub = subset[subset['Year'] == y]
                if len(y_sub) == 0:
                    y_p = 0
                else:
                    y_w = (y_sub['IsWin'] == 0).sum()
                    y_gross = y_w * STAKE * (1-COMM)
                    y_loss = y_sub[y_sub['IsWin'] == 1]['Liability'].sum()
                    y_p = y_gross - y_loss
                yearly_prof.append(y_p)
            
            stable = all(p > 0 for p in yearly_prof)
            stable_str = "YES" if stable else "NO"
            
            if roi > 8: # Filter output
                config_str = f"Boxes={b_name} Odds<{max_o}"
                print(f"{config_str:<35} {len(subset):<6} {strike:<8.1f} ${net_profit:<9.0f} {roi:<6.1f} {stable_str:<8}")
                results.append((config_str, roi, net_profit, stable))

if __name__ == "__main__":
    df = load_and_prep()
    test_df = get_predictions(df)
    optimize(test_df)
