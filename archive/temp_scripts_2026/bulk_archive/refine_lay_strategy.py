
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_prod.json"
COMMISSION = 0.08

def get_data():
    print("[1/3] Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    
    # Same query as production
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, ge.Box as RawBox, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Price5Min, ge.Weight,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP > 0 AND ge.FinishTime > 0
    AND rm.MeetingDate >= date('now', '-12 months')
    """
    df = pd.read_sql_query(query, conn)
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # Clean Margin1 (Critical Fix)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0)
    
    return df

def feature_engineering(df):
    print("[2/3] Engineering Features...")
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    for col in ['Place', 'StartPrice', 'RunTime', 'SplitMargin', 'Distance']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Lags
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)
            
    # Averages
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime']
    df['RunSpeed_avg'] = df.groupby('GreyhoundID')['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    
    # Categorical
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
        
    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df, feature_cols

def run_grid_search(df, feature_cols):
    print("[3/3] Predicting & Grid Seeking...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    
    # Grid Parameters - FOCUSED ON LOW VOLUME / HIGH ROI
    max_prices = [1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
    max_probs = [0.05, 0.10, 0.15, 0.20]
    max_edges = [-0.50, -0.60, -0.70, -0.80]
    
    results = []
    
    print("\n" + "="*80)
    print(f"{'MAX PRICE':<10} | {'MAX PROB':<10} | {'MAX EDGE':<10} | {'BETS':<6} | {'STRIKE':<6} | {'ROI %':<8} | {'PROFIT':<10}")
    print("="*80)
    
    for max_p in max_prices:
        for prob_cap in max_probs:
            for edge_cap in max_edges:
                
                # Filter Logic
                # Edge = Prob - Implied (Negative for Lay)
                # Lay Criteria: Price < Max, Prob < Cap, Edge < Cap
                # HISTORY FILTER: Must have at least 2 previous races (Place_Lag2 != -1)
                # We only want to lay dogs we have DATA on.
                
                mask = (df['StartPrice'] > 1.01) & \
                       (df['StartPrice'] < max_p) & \
                       (df['Prob_V33'] < prob_cap) & \
                       ((df['Prob_V33'] - (1/df['StartPrice'])) < edge_cap) & \
                       (df['Place_Lag2'] != -1)
                
                subset = df[mask].copy()
                
                if len(subset) < 10: continue 
                
                bets = len(subset)
                lay_wins = len(subset[subset['Place'] > 1])
                strike = (lay_wins / bets) * 100
                
                # Metric: Target Profit $10
                # If we win (dog loses), we get $10 * (1-Comm)
                # If we lose (dog wins), we pay $10
                
                target_profit = 10.0
                subset['profit'] = np.where(
                    subset['Place'] > 1,
                    target_profit * (1 - COMMISSION), 
                    -target_profit
                )
                
                total_profit = subset['profit'].sum()
                total_liability = bets * 10.0 # Risked $10 every time
                roi = (total_profit / total_liability) * 100
                
                results.append({
                    'MaxPrice': max_p,
                    'MaxProb': prob_cap,
                    'MaxEdge': edge_cap,
                    'Bets': bets,
                    'Strike': strike,
                    'ROI': roi,
                    'Profit': total_profit
                })

    results_df = pd.DataFrame(results).sort_values('ROI', ascending=False)
    
    for i, row in results_df.head(30).iterrows():
         print(f"${row['MaxPrice']:<9.2f} | {row['MaxProb']:<9.2f} | {row['MaxEdge']:<9.2f} | {int(row['Bets']):<6} | {row['Strike']:<6.1f} | {row['ROI']:<8.2f} | ${row['Profit']:<9.2f}")

if __name__ == "__main__":
    df = get_data()
    df, features = feature_engineering(df)
    run_grid_search(df, features)
