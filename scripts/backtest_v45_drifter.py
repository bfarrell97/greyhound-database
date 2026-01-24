import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def backtest_v45_drifter():
    print("Loading Data (2024-2025) for V45 Drifter Backtest...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, 
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND t.TrackName NOT IN ('LAUNCESTON', 'HOBART', 'DEVONPORT')
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Base Features (V41)
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # 2. Lag Features (Rolling Drift Context)
    print("Engineering Historical Drifter Features...")
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Drifter'] = (df_clean['MoveRatio'] < 0.95).astype(int)

    df_clean = df_clean.sort_values('MeetingDate')
    
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 3. Base Probabilities
    print("Generating Base Model Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 4. Predict using Test Model
    print("Loading V45 TEST Model...")
    model_v45 = joblib.load('models/xgb_v45_test.pkl')
    
    features_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    print("Generating Predictions...")
    X_test = df_clean[features_v45]
    probs = model_v45.predict_proba(X_test)[:, 1]
    df_clean['Drift_Prob'] = probs
    
    # 5. Evaluate Thresholds & Odds Bands
    target_thresholds = [0.60, 0.63, 0.65]
    price_bands = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    
    for t in target_thresholds:
        print(f"\nAnalyzing Threshold >= {t} (Cap $30)...")
        print(f"{'Band':<12} | {'Count':<6} | {'Precision':<10} | {'ROI (Lay)':<12} | {'Avg Price':<10}")
        print("-" * 60)
        
        # Filter by threshold first
        mask_t = (df_clean['Drift_Prob'] >= t) & (df_clean['Price5Min'] < 30.0)
        subset = df_clean[mask_t].copy()
        
        if len(subset) == 0:
            print("No bets found.")
            continue
            
        subset['Dog_Win'] = (pd.to_numeric(subset['Position'], errors='coerce') == 1).astype(int)
        subset['PnL'] = np.where(subset['Dog_Win'] == 1, -10 * (subset['Price5Min'] - 1), 10 * 0.95)
        
        for low, high in price_bands:
            band_mask = (subset['Price5Min'] >= low) & (subset['Price5Min'] < high)
            band_data = subset[band_mask]
            
            if len(band_data) == 0:
                print(f"${low:<2}-${high:<2}      | 0      | N/A       | N/A          | N/A")
                continue
                
            profit = band_data['PnL'].sum()
            roi = (profit / (len(band_data) * 10)) * 100
            prec = (band_data['Is_Drifter'] == 1).mean()
            avg_p = band_data['Price5Min'].mean()
            
            print(f"${low:<2}-${high:<2}      | {len(band_data):<6} | {prec:>9.1%} | {roi:>+11.1f}% | ${avg_p:>9.2f}")
            
        # Overall for this threshold
        total_profit = subset['PnL'].sum()
        total_roi = (total_profit / (len(subset) * 10)) * 100
        print("-" * 60)
        print(f"{'TOTAL':<12} | {len(subset):<6} | {(subset['Is_Drifter']==1).mean():>9.1%} | {total_roi:>+11.1f}% |")

    print("\nROI = Profit / (Stake * Count). Stake $10.")

if __name__ == "__main__":
    backtest_v45_drifter()
