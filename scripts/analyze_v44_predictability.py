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

def analyze_predictability():
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
    
    # Target Definition (Ground Truth)
    # Steam is dropping price. Ratio = P5 / BSP.
    # Target Definition (Ground Truth)
    # Steam is dropping price. Ratio = P5 / BSP.
    # Feature Definition: MUST BE > 1.15 (Match Training)
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    
    # 2. Lag Features
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
    
    # Test Set Only
    df_test = df_clean[df_clean['MeetingDate'] >= '2024-01-01'].copy()
    
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    probs = model.predict_proba(df_test[features_v44])[:, 1]
    df_test['PredProb'] = probs
    
    # 4. REPORT
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    
    print("\n" + "="*100)
    print("V44 MODEL ANALYSIS: PREDICTABILITY vs VOLUME (2024-2025)")
    print("Config: Price < $30.00 ONLY | Eval: MoveRatio > 1.0 (Beat BSP)")
    print("="*100)
    print(f"{'Thres >':<8} | {'Volume':<8} | {'Precision':<10} | {'Win Rate':<10} | {'ROI %':<8} | {'Avg Price':<10}")
    print("-" * 100)
    
    for t in thresholds:
        # Filter Bets: Prob > t AND Price < 30
        subset = df_test[(df_test['PredProb'] > t) & (df_test['Price5Min'] < 30.0)].copy()
        count = len(subset)
        
        if count == 0:
            print(f"{t:<8.2f} | 0        | -          | -          | -        | -")
            continue
            
        # Precision: How many ACTUALLY steamed?
        # Precision: How many Beat BSP? (Eval Target > 1.0)
        beat_bsp = (subset['MoveRatio'] > 1.0).sum()
        precision = (beat_bsp / count) * 100
        
        # Win Rate: How many won the race?
        subset['Win'] = (pd.to_numeric(subset['Position'], errors='coerce') == 1).astype(int)
        win_rate = subset['Win'].mean() * 100
        
        # ROI
        # PnL = (Price - 1)*0.95 if Win, else -1
        pnl = np.where(subset['Win']==1, (subset['Price5Min']-1)*0.95, -1.0).sum()
        roi = pnl / count * 100
        
        avg_price = subset['Price5Min'].mean()
        
        print(f"{t:<8.2f} | {count:<8} | {precision:>9.1f}% | {win_rate:>9.1f}% | {roi:>7.2f}% | ${avg_price:>8.2f}")
        
    print("="*100)

if __name__ == "__main__":
    analyze_predictability()
