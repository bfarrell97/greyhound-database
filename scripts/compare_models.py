"""
Compare Models - Full Comparison (Back + Lay)
Tests V44 Back Model and V45 Lay Model on Nov 2025 data with proper feature engineering.
"""
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

def compare_models():
    print("--- FULL MODEL COMPARISON (Back + Lay) ---")
    
    # Load data from 2024 to ensure proper rolling features
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
    
    print(f"Loaded {len(df)} rows (2024-Present)")
    
    # 1. Base Features (V41)
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    df_clean = df_clean.sort_values('MeetingDate')
    
    # 2. Targets
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer_Hist'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    df_clean['Is_Drifter_Hist'] = (df_clean['MoveRatio'] < 0.95).astype(int)
    
    # 3. Rolling Features (Steamer)
    print("Engineering Steam Lags...")
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 4. Rolling Features (Drifter)
    print("Engineering Drift Lags...")
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 5. Base Probabilities
    print("Generating V41 Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 6. Load Models
    print("Loading Models...")
    # Back Models
    back_test = joblib.load('models/xgb_v44_steamer.pkl')
    back_prod = joblib.load('models/xgb_v44_production.pkl')
    # Lay Models
    lay_test = joblib.load('models/xgb_v45_test.pkl')
    lay_prod = joblib.load('models/xgb_v45_production.pkl')
    
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    features_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    # 7. Filter to Nov 2025 AFTER feature engineering
    df_nov = df_clean[(df_clean['MeetingDate'] >= '2025-11-01') & 
                      (df_clean['MeetingDate'] <= '2025-11-30')].copy()
    
    print(f"Nov 2025 Test Set: {len(df_nov)} rows")
    
    # 8. Predictions
    X_back = df_nov[features_v44]
    X_lay = df_nov[features_v45]
    
    df_nov['Back_Test_Prob'] = back_test.predict_proba(X_back)[:, 1]
    df_nov['Back_Prod_Prob'] = back_prod.predict_proba(X_back)[:, 1]
    df_nov['Lay_Test_Prob'] = lay_test.predict_proba(X_lay)[:, 1]
    df_nov['Lay_Prod_Prob'] = lay_prod.predict_proba(X_lay)[:, 1]
    
    # 9. Metrics
    # Beat BSP = Price5Min > BSP (Steamer for BACK)
    df_nov['Beat_BSP'] = (df_nov['Price5Min'] > df_nov['BSP']).astype(int)
    # Is Drifter = Price5Min < BSP * 0.95 (for LAY success)
    df_nov['Is_Drifter'] = (df_nov['MoveRatio'] < 0.95).astype(int)
    
    price_mask = df_nov['Price5Min'] < 30.0
    
    # --- BACK MODEL ---
    baseline_back = df_nov['Beat_BSP'].mean()
    print(f"\n=== BACK MODEL (V44) ===")
    print(f"Baseline Beat BSP Rate (Nov): {baseline_back:.1%}")
    print(f"{'Threshold':<10} | {'TEST Vol':<8} {'BeatBSP%':<8} | {'PROD Vol':<8} {'BeatBSP%':<8}")
    print("-" * 70)
    
    for t in [0.30, 0.32, 0.35, 0.38, 0.40, 0.45]:
        mask_test = (df_nov['Back_Test_Prob'] >= t) & price_mask
        n_test = mask_test.sum()
        beat_test = df_nov.loc[mask_test, 'Beat_BSP'].mean() if n_test > 0 else 0.0
        
        mask_prod = (df_nov['Back_Prod_Prob'] >= t) & price_mask
        n_prod = mask_prod.sum()
        beat_prod = df_nov.loc[mask_prod, 'Beat_BSP'].mean() if n_prod > 0 else 0.0
        
        print(f">= {t:<8} | {n_test:<8} {beat_test:>7.1%} | {n_prod:<8} {beat_prod:>7.1%}")
    
    # --- LAY MODEL ---
    baseline_lay = df_nov['Is_Drifter'].mean()
    print(f"\n=== LAY MODEL (V45) ===")
    print(f"Baseline Drifter Rate (Nov): {baseline_lay:.1%}")
    print(f"{'Threshold':<10} | {'TEST Vol':<8} {'Drift%':<8} | {'PROD Vol':<8} {'Drift%':<8}")
    print("-" * 70)
    
    for t in [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        mask_test = (df_nov['Lay_Test_Prob'] >= t) & price_mask
        n_test = mask_test.sum()
        drift_test = df_nov.loc[mask_test, 'Is_Drifter'].mean() if n_test > 0 else 0.0
        
        mask_prod = (df_nov['Lay_Prod_Prob'] >= t) & price_mask
        n_prod = mask_prod.sum()
        drift_prod = df_nov.loc[mask_prod, 'Is_Drifter'].mean() if n_prod > 0 else 0.0
        
        print(f">= {t:<8} | {n_test:<8} {drift_test:>7.1%} | {n_prod:<8} {drift_prod:>7.1%}")

if __name__ == "__main__":
    compare_models()
