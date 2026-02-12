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

MODEL_PATH = 'models/xgb_v44_steamer.pkl'

def run_backtest_v44_fast():
    print(f"Loading Model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    print("Loading Data (2024-2025)...")
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
    WHERE rm.MeetingDate >= '2023-01-01'
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Base Features
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # Target
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    
    # 2. Lag Features (Recalculate to match training)
    print("Engineering Steam Lags...")
    df_clean = df_clean.sort_values('MeetingDate')
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 3. Probabilities
    print("Generating Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
    
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # Filter for Backtest Period (2024-2025)
    df_bt = df_clean[df_clean['MeetingDate'] >= '2024-01-01'].copy()
    
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    probs = model.predict_proba(df_bt[features_v44])[:, 1]
    df_bt['PredProb'] = probs
    
    # Calculate Beat BSP for Precision
    df_bt['BeatBSP'] = (df_bt['Price5Min'] / df_bt['BSP'] > 1.0).astype(int)

    thresholds = [0.30, 0.35, 0.40, 0.45]

    for t in thresholds:
        print("\n" + "="*110)
        print(f"V44 BACKTEST REPORT (2024-2025) - ODDS BANDS BREAKDOWN")
        print(f"Config: Threshold > {t:.2f} | Precision = Beat BSP %")
        print("="*110)
        print(f"{'Price Band':<15} | {'Bets':<6} | {'Precision':<10} | {'Win %':<8} | {'Avg $':<8} | {'PnL ($)':<10} | {'ROI %':<8}")
        print("-" * 110)
        
        bands = [
            (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15), (15, 20), (20, 30)
        ]
        
        total_bets = 0
        total_pnl = 0
        
        for low, high in bands:
            # Bets > t Prob AND Price in Band
            mask = (df_bt['PredProb'] > t) & (df_bt['Price5Min'] >= low) & (df_bt['Price5Min'] < high)
            bets = df_bt[mask].copy()
            
            count = len(bets)
            if count == 0: 
                print(f"${low:<4}-${high:<4} | 0      | -          | -        | -        | 0.00       | 0.00")
                continue
            
            bets['Win'] = (pd.to_numeric(bets['Position'], errors='coerce') == 1).astype(int)
            win_rate = bets['Win'].mean() * 100
            avg_price = bets['Price5Min'].mean()
            precision = bets['BeatBSP'].mean() * 100
            
            # PnL (Fixed Stake)
            pnl = np.where(bets['Win']==1, (bets['Price5Min']-1)*0.95, -1.0).sum()
            roi = pnl / count * 100
            
            label = f"${low}-${high}"
            print(f"{label:<15} | {count:<6} | {precision:>9.1f}% | {win_rate:>8.1f} | {avg_price:>8.2f} | {pnl:>10.2f} | {roi:>8.2f}")
            
            total_bets += count
            total_pnl += pnl
            
        print("-" * 110)
        print(f"{'TOTAL':<15} | {total_bets:<6} | {'-':<10} | {'-':<8} | {'-':<8} | {total_pnl:>10.2f} | {total_pnl/total_bets*100:>8.2f}")
        print("="*110)

if __name__ == "__main__":
    run_backtest_v44_fast()
