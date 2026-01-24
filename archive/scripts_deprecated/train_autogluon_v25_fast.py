
import sqlite3
import pandas as pd
import numpy as np
import time
from autogluon.tabular import TabularPredictor
import os

# Configuration
DB_PATH = 'greyhound_racing.db'
MODEL_PATH = 'models/autogluon_margin_v25_fast'
TRAIN_ROWS = 2000000

def main():
    print("="*70)
    print("V25 MARGIN MODEL (FAST VECTORIZED)")
    print("="*70)
    
    start_time = time.time()
    
    conn = sqlite3.connect(DB_PATH)
    print("Loading Data...")
    
    # Load Benchmarks for NormTime
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = bench_df.set_index(['TrackName', 'Distance'])['MedianTime'].to_dict()
    
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box, ge.Margin,
           ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
           r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
      AND ge.Position IS NOT NULL
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Preprocessing
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    
    # --- CRITICAL FIX: Winner Margin = 0 ---
    # DB has "7.00L" -> Strip 'L'
    df['Margin'] = df['Margin'].astype(str).str.replace('L', '', regex=False).str.strip()
    df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce').fillna(99)
    df.loc[df['Position'] == 1, 'Margin'] = 0.0
    
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
    # Benchmarks
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    
    print(f"Loaded {len(df):,} rows in {time.time() - start_time:.1f}s")
    
    # --- VECTORIZED FEATURE GENERATION ---
    print("Generating Features (Vectorized)...")
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # GroupBy object
    g = df.groupby('GreyhoundID')
    
    # 1. Margin Features
    df['Margin_Mean_3'] = g['Margin'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['Margin_Mean_10'] = g['Margin'].transform(lambda x: x.shift(1).rolling(10).mean())
    df['LastMargin'] = g['Margin'].shift(1)
    df['Margin_Std_10'] = g['Margin'].transform(lambda x: x.shift(1).rolling(10).std())
    df['Margin_Trend'] = df['LastMargin'] - df['Margin_Mean_3']
    
    # 2. Position Features
    df['Pos_Mean_5'] = g['Position'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['LastPos'] = g['Position'].shift(1)
    df['WinRate_10'] = g['Position'].transform(lambda x: x.shift(1).rolling(10).apply(lambda s: (s==1).mean()))
    
    # 3. Time Features
    df['Time_Mean_5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['Time_Best_10'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(10).min())
    df['LastTime'] = g['NormTime'].shift(1)
    
    # 4. Speed (Beyer)
    df['Beyer_Mean_5'] = g['BeyerSpeedFigure'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['LastBeyer'] = g['BeyerSpeedFigure'].shift(1)
    
    # 5. Split
    df['Split_Mean_5'] = g['Split'].transform(lambda x: x.shift(1).rolling(5).mean())
    
    # Fill NA (First runs)
    features = [
        'Margin_Mean_3', 'Margin_Mean_10', 'LastMargin', 'Margin_Std_10', 'Margin_Trend',
        'Pos_Mean_5', 'LastPos', 'WinRate_10',
        'Time_Mean_5', 'Time_Best_10', 'LastTime',
        'Beyer_Mean_5', 'LastBeyer',
        'Split_Mean_5'
    ]
    df[features] = df[features].fillna(0)
    
    # Clean Infinite
    df = df.replace([np.inf, -np.inf], 0)
    
    # --- TRAIN/TEST SPLIT ---
    # Training: Before 2024-06-01
    # Testing: After 2024-06-01
    train_data = df[df['MeetingDate'] < '2024-06-01']
    test_data = df[df['MeetingDate'] >= '2024-06-01']
    
    print(f"Training on {len(train_data):,} rows")
    print(f"Testing on {len(test_data):,} rows")
    
    # --- AUTOGLUON ---
    print("Training AutoGluon...")
    predictor = TabularPredictor(
        label='Margin', 
        path=MODEL_PATH, 
        problem_type='regression',
        eval_metric='mean_absolute_error'
    ).fit(
        train_data[features + ['Margin']],
        time_limit=600,
        presets='medium_quality'
    )
    
    # --- EVALUATION ---
    print("\n" + "="*70)
    print("ACCURACY REPORT")
    print("="*70)

    # 1. Margin Accuracy
    test_data['PredMargin'] = predictor.predict(test_data[features])
    margin_mae = np.mean(np.abs(test_data['Margin'] - test_data['PredMargin']))
    
    # Margin MAPE (Exclude Winners where Margin=0 to avoid inf)
    non_winners = test_data[test_data['Margin'] > 0.01].copy()
    if len(non_winners) > 0:
        margin_mape = np.mean(np.abs((non_winners['Margin'] - non_winners['PredMargin']) / non_winners['Margin'])) * 100
    else:
        margin_mape = 0
        
    print(f"MARGIN ACCURACY:")
    print(f"  MAE:  {margin_mae:.3f} lengths")
    print(f"  MAPE: {margin_mape:.2f}% (Non-Winners)")

    # 2. Price Accuracy (Rated Price vs BSP)
    print("\nPRICE ACCURACY (Rated Price vs BSP):")
    
    # Calculate Probabilities (Softmin per Race)
    def calculate_probs(group):
        scores = np.exp(-1.0 * group['PredMargin'])
        return scores / scores.sum()
    
    test_data['RaceID'] = test_data['RaceID'].astype(int)
    probs = test_data.groupby('RaceID').apply(calculate_probs).reset_index(level=0, drop=True)
    
    # Align indices carefully
    # groupby().apply() returns a Series with index matching original if done right, 
    # but sometimes multi-index. 
    # Logic: groupby preserves index.
    
    test_data['PredProb'] = probs
    test_data['RatedPrice'] = 1 / test_data['PredProb']

    # Filter valid BSP for fair comparison
    valid = test_data[(test_data['BSP'] > 1) & (test_data['BSP'] < 50)].copy()
    
    # Error Metrics
    valid['Error'] = np.abs(valid['RatedPrice'] - valid['BSP'])
    valid['PctError'] = valid['Error'] / valid['BSP']
    
    mape = valid['PctError'].mean() * 100
    corr = valid[['RatedPrice', 'BSP']].corr().iloc[0,1]
    
    print(f"  Evaluated on {len(valid):,} rows (BSP < 50)")
    print(f"  MAPE (Rated vs BSP): {mape:.2f}%")
    print(f"  Correlation: {corr:.4f}")
    
    # Breakdown by Odds Range
    print("\nAccuracy by BSP Range:")
    ranges = [(0,5), (5,10), (10,20), (20,50)]
    for low, high in ranges:
        subset = valid[(valid['BSP'] >= low) & (valid['BSP'] < high)]
        if len(subset) > 0:
            sub_mape = subset['PctError'].mean() * 100
            print(f"  ${low}-${high}: MAPE {sub_mape:.2f}% ({len(subset)} rows)")

if __name__ == "__main__":
    main()
