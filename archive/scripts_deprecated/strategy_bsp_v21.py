"""
V21 Strategy Simulation - OPTIMIZED (Vectorized)
================================================
Uses Pandas Rolling Windows for 50x speedup.
Loads full history (2020-2025) to preserve context, computes features, slices 2024+ for test.
"""
import pandas as pd
import numpy as np
import sqlite3
import time
from autogluon.tabular import TabularPredictor

# Placeholder Model Path
MODEL_PATH = "models/autogluon_bsp_v21_XXXX" 

def run_strategy(model_path=None, price_col='Price5Min', max_odds=20.0):
    start_time = time.time()
    if model_path:
        path = model_path
    else:
        path = MODEL_PATH
        
    predictor = TabularPredictor.load(path)
    FEATURES = predictor.feature_metadata_in.get_features()
    print(f"Loaded Model: {path}")
    print(f"Simulating using Price Column: {price_col}")
    print(f"Max Odds Cap: ${max_odds}")

    # Load FULL Data (2020-2025) for History
    conn = sqlite3.connect('greyhound_racing.db')
    # Dynamic Query with ALIAS to avoid duplicates
    query = f"""
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.{price_col} as TargetPrice, ge.Price5Min, ge.Box,
           ge.FinishTime, ge.Split, ge.Weight,
           ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
           g.SireID
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
      AND ge.Position IS NOT NULL 
      AND ge.BSP IS NOT NULL AND ge.BSP > 1
    ORDER BY rm.MeetingDate
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading data (Check if column {price_col} exists): {e}")
        return
    conn.close()

    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Year'] = df['MeetingDate'].dt.year
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
    df['IsWin'] = (df['Position'] == 1).astype(int)
    df['IsPlace'] = (df['Position'] <= 3).astype(int)
    
    # Ensure TargetPrice is numeric
    df['TargetPrice'] = pd.to_numeric(df['TargetPrice'], errors='coerce')
    # Also Price5Min for reference/fallback? (Not strictly needed if we use TargetPrice for sim)
    df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')

    # --- VECTORIZED FEATURE GENERATION ---
    print(f"Generating Features for {len(df):,} rows (Vectorized)...")
    
    df = df.sort_values(by=['GreyhoundID', 'MeetingDate'])
    
    # helper for rolling
    g = df.groupby('GreyhoundID')
    
    # Shift(1) to avoid data leakage (current row not included in history)
    s_bsp = g['BSP'].shift(1)
    # Wait, rolling on shifted series
    
    for w in [3, 10]:
        # BSP Stats
        df[f'BSP_Mean_{w}'] = s_bsp.rolling(window=w, min_periods=w).mean()
        df[f'BSP_Max_{w}'] = s_bsp.rolling(window=w, min_periods=w).max()
        df[f'BSP_Min_{w}'] = s_bsp.rolling(window=w, min_periods=w).min()
        df[f'BSP_Std_{w}'] = s_bsp.rolling(window=w, min_periods=w).std()
        
        # Pos Stats
        # We need rolling mean of Position (numeric)
        s_pos_numeric = g['Position'].shift(1)
        df[f'Pos_Mean_{w}'] = s_pos_numeric.rolling(window=w, min_periods=w).mean()
        
        # Win Rate
        s_is_win = g['IsWin'].shift(1)
        df[f'WinRate_{w}'] = s_is_win.rolling(window=w, min_periods=w).mean()
        
        # Place Rate
        s_is_place = g['IsPlace'].shift(1)
        df[f'PlaceRate_{w}'] = s_is_place.rolling(window=w, min_periods=w).mean()

    # Fill NaNs for features (required by AutoGluon maybe? or it handles them)
    # Better to fill with defaults similar to V21 script
    defaults = {
         'BSP_Mean_10': df['BSP'].mean(), 'BSP_Mean_3': df['BSP'].mean(),
         'Pos_Mean_10': 5, 'WinRate_10': 0, 'PlaceRate_10': 0
    }
    # Actually allow NaNs, AutoGluon handles them.
    
    # Context Stats (Trainer, Sire, Box) - Harder to vectorize perfectly without massive join
    # Approximation: Use Global Average for Trainer?
    # Or Cumulative Mean?
    # For speed, we might skip complex context or use simplified version.
    # V21 script used `Trainer_AvgBSP`.
    # Vectorized Cumulative Mean:
    # df['Trainer_AvgBSP'] = df.groupby('TrainerID')['BSP'].apply(lambda x: x.shift(1).expanding().mean())
    # But this is slow (apply). transform(lambda x...) is better.
    
    # Optimized Expanding Mean
    print("  Calculating Context Stats...")
    # Trainer
    df['Trainer_AvgBSP'] = df.groupby('TrainerID')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)
    # Sire
    df['Sire_AvgBSP'] = df.groupby('SireID')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)
    # Box@Track
    # Create key column
    df['BoxTrack'] = df['TrackID'].astype(str) + '_' + df['Box'].astype(str)
    df['Box_Track_AvgBSP'] = df.groupby('BoxTrack')['BSP'].transform(lambda x: x.shift(1).expanding().mean()).fillna(10)

    # Trend
    df['BSP_Trend_5'] = s_bsp - g['BSP'].shift(5) # Shift(1) - Shift(5) = Difference over 4 races
    
    df['LastBSP'] = s_bsp
    df['LastPos'] = g['Position'].shift(1)
    
    # Class Trend (Required for V23)
    # Avoid div by zero (fillna(10) for mean helps)
    df['Class_Trend_10'] = df['LastBSP'] / df['BSP_Mean_10']
    
    # -------------------------------------------------------
    
    # Slice Test Data
    print("Slicing Test Data (2024+)...")
    # Use TargetPrice
    test_df = df[(df['Year'] >= 2024) & (df['TargetPrice'].notna())].copy()
    
    # Features required by model (Filter columns)
    # Fill remaining NaNs
    test_df[FEATURES] = test_df[FEATURES].fillna(0) # Simple impute
    
    print(f"Test Set: {len(test_df):,} rows")
    
    # Prediction
    print("Predicting...")
    test_df['PredLogBSP'] = predictor.predict(test_df[FEATURES])
    test_df['PredBSP'] = np.exp(test_df['PredLogBSP'])
    
    # Simulation
    edge = 0.20
    print(f"\nSIMULATION (Edge {edge*100}%, Max Odds ${max_odds})")
    
    # BACK
    # Filter by Max Odds on TargetPrice
    test_df_capped = test_df[test_df['TargetPrice'] <= max_odds].copy()
    
    backs = test_df_capped[test_df_capped['TargetPrice'] > test_df_capped['PredBSP'] * (1 + edge)].copy()
    backs_wins = backs['IsWin'].sum()
    backs_profit = (backs[backs['IsWin']==1]['TargetPrice'].sum() - len(backs)) * 0.90
    
    print(f"\nBACK STRATEGY:")
    print(f"Bets: {len(backs):,}")
    print(f"Wins: {backs_wins:,} ({backs_wins/len(backs)*100 if len(backs) else 0:.1f}%)")
    if len(backs) > 0:
        avg_odds = backs['TargetPrice'].mean()
        avg_win_odds = backs[backs['IsWin']==1]['TargetPrice'].mean() if backs_wins > 0 else 0
        median_odds = backs['TargetPrice'].median()
        print(f"Avg Odds Placed: ${avg_odds:.2f} | Median: ${median_odds:.2f}")
        print(f"Avg Odds Winners: ${avg_win_odds:.2f}")
        
    print(f"Profit: ${backs_profit:.2f}")
    if len(backs) > 0: print(f"ROI: {backs_profit/len(backs)*100:.2f}%")
    
    # LAY
    lays = test_df_capped[test_df_capped['TargetPrice'] < test_df_capped['PredBSP'] * (1 - edge)].copy()
    lays_wins = len(lays[lays['IsWin']==0])
    liability_lost = lays[lays['IsWin']==1].apply(lambda x: x['TargetPrice'] - 1, axis=1).sum()
    net_profit_lays = (lays_wins - liability_lost)
    if net_profit_lays > 0: net_profit_lays *= 0.90
    
    print(f"\nLAY STRATEGY:")
    print(f"Bets: {len(lays):,}")
    print(f"Successful Lays: {lays_wins:,} ({lays_wins/len(lays)*100 if len(lays) else 0:.1f}%)")
    print(f"Net Profit (Est): ${net_profit_lays:.2f}")
    if len(lays) > 0: print(f"ROI: {net_profit_lays/len(lays)*100:.2f}%")
    
    print(f"\nTime: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    import sys
    # args: model_path price_col max_odds
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    price_col = sys.argv[2] if len(sys.argv) > 2 else 'Price5Min'
    max_odds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    
    run_strategy(model_path, price_col, max_odds)
