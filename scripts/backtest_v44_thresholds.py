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

def backtest_v44_thresholds():
    print("Loading Data (2024-07-01 onwards) for V44/V45 Backtest with Conflict Exclusion...")
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
    WHERE rm.MeetingDate >= '2024-07-01'
    AND t.TrackName NOT IN ('LAUNCESTON', 'HOBART', 'DEVONPORT')
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} entries")
    
    # 1. Base Features (V41)
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # 2. Lag Features (Rolling Steam Context)
    print("Engineering Historical Steamer Features...")
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer_Hist'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    
    df_clean = df_clean.sort_values('MeetingDate')
    
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 3. Base Probabilities
    print("Generating V41 Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 4. Predict using V44 BACK Model
    print("Loading V44 Steamer Model (BACK)...")
    model_v44 = joblib.load('models/xgb_v44_steamer.pkl')
    
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    print("Generating V44 BACK Predictions...")
    X_test = df_clean[features_v44]
    df_clean['Steam_Prob'] = model_v44.predict_proba(X_test)[:, 1]
    
    # 5. Predict using V45 LAY Model
    print("Loading V45 Drift Model (LAY)...")
    try:
        model_v45 = joblib.load('models/xgb_v45_production.pkl')
        print("Generating V45 LAY Predictions...")
        df_clean['Drift_Prob'] = model_v45.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"[WARN] V45 Model load failed: {e}")
        # Fallback: Use inverse of Steam_Prob as proxy for Drift
        df_clean['Drift_Prob'] = 1 - df_clean['Steam_Prob']
    
    # Add dog win flag
    df_clean['Dog_Win'] = (pd.to_numeric(df_clean['Position'], errors='coerce') == 1).astype(int)
    
    # =========================================================================
    # CONFLICT EXCLUSION BACKTEST
    # =========================================================================
    print("\n" + "="*75)
    print("CONFLICT EXCLUSION BACKTEST")
    print("="*75)
    print("Testing if excluding bets when opposite model shows potential improves ROI")
    
    # Configuration
    BACK_THRESHOLD = 0.30  # Steam_Prob >= this = BACK signal
    LAY_THRESHOLD = 0.55   # Drift_Prob >= this = LAY signal
    PRICE_CAP = 30.0
    EXCLUSION_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]
    
    # --- BACK SIGNALS ---
    print(f"\n### BACK SIGNALS (Steam_Prob >= {BACK_THRESHOLD}, Price < ${PRICE_CAP}) ###")
    print("-" * 75)
    
    # Control: No exclusion
    mask_back = (df_clean['Steam_Prob'] >= BACK_THRESHOLD) & (df_clean['Price5Min'] < PRICE_CAP)
    control_back = df_clean[mask_back].copy()
    
    # Calculate BACK PnL: Win = (Price-1), Lose = -1 (unit stake)
    control_back['PnL'] = np.where(control_back['Dog_Win'] == 1, control_back['Price5Min'] - 1, -1)
    
    if len(control_back) > 0:
        sr = control_back['Dog_Win'].mean() * 100
        pnl = control_back['PnL'].sum()
        roi = pnl / len(control_back) * 100
        print(f"CONTROL (no exclusion): {len(control_back):,} bets | SR: {sr:.1f}% | PnL: {pnl:+.2f} | ROI: {roi:+.2f}%")
    
    # Test: Exclude where LAY prob is high
    print(f"\nTEST (exclude if Drift_Prob >= threshold):")
    for thresh in EXCLUSION_THRESHOLDS:
        mask_excl = mask_back & (df_clean['Drift_Prob'] < thresh)
        test = df_clean[mask_excl].copy()
        
        if len(test) > 0:
            test['PnL'] = np.where(test['Dog_Win'] == 1, test['Price5Min'] - 1, -1)
            sr = test['Dog_Win'].mean() * 100
            pnl = test['PnL'].sum()
            roi = pnl / len(test) * 100
            excl = len(control_back) - len(test)
            print(f"  Thresh {thresh:.2f}: {len(test):,} bets (-{excl}) | SR: {sr:.1f}% | PnL: {pnl:+.2f} | ROI: {roi:+.2f}%")
    
    # --- LAY SIGNALS ---
    print(f"\n### LAY SIGNALS (Drift_Prob >= {LAY_THRESHOLD}, Price < ${PRICE_CAP}) ###")
    print("-" * 75)
    
    # Control: No exclusion
    mask_lay = (df_clean['Drift_Prob'] >= LAY_THRESHOLD) & (df_clean['Price5Min'] < PRICE_CAP)
    control_lay = df_clean[mask_lay].copy()
    
    # Calculate LAY PnL: Win (dog loses) = +1, Lose (dog wins) = -(Price-1)
    control_lay['PnL'] = np.where(control_lay['Dog_Win'] == 0, 1, -(control_lay['Price5Min'] - 1))
    
    if len(control_lay) > 0:
        sr = (control_lay['Dog_Win'] == 0).mean() * 100  # LAY wins when dog doesn't win
        pnl = control_lay['PnL'].sum()
        roi = pnl / len(control_lay) * 100
        print(f"CONTROL (no exclusion): {len(control_lay):,} bets | SR: {sr:.1f}% | PnL: {pnl:+.2f} | ROI: {roi:+.2f}%")
    
    # Test: Exclude where BACK prob is high
    print(f"\nTEST (exclude if Steam_Prob >= threshold):")
    for thresh in EXCLUSION_THRESHOLDS:
        mask_excl = mask_lay & (df_clean['Steam_Prob'] < thresh)
        test = df_clean[mask_excl].copy()
        
        if len(test) > 0:
            test['PnL'] = np.where(test['Dog_Win'] == 0, 1, -(test['Price5Min'] - 1))
            sr = (test['Dog_Win'] == 0).mean() * 100
            pnl = test['PnL'].sum()
            roi = pnl / len(test) * 100
            excl = len(control_lay) - len(test)
            print(f"  Thresh {thresh:.2f}: {len(test):,} bets (-{excl}) | SR: {sr:.1f}% | PnL: {pnl:+.2f} | ROI: {roi:+.2f}%")
    
    # --- SUMMARY ---
    print("\n" + "="*75)
    print("SUMMARY")
    print("="*75)
    
    if len(control_back) > 0 and len(control_lay) > 0:
        control_back_pnl = np.where(control_back['Dog_Win'] == 1, control_back['Price5Min'] - 1, -1).sum()
        control_lay_pnl = np.where(control_lay['Dog_Win'] == 0, 1, -(control_lay['Price5Min'] - 1)).sum()
        combined_pnl = control_back_pnl + control_lay_pnl
        combined_bets = len(control_back) + len(control_lay)
        combined_roi = combined_pnl / combined_bets * 100
        
        print(f"Combined CONTROL: {combined_bets:,} bets | PnL: {combined_pnl:+.2f} | ROI: {combined_roi:+.2f}%")
    
    print("\nNote: Exclusion helps if it filters out losing bets while keeping winners.")
    print("Look for thresholds that improve ROI significantly vs CONTROL.")

if __name__ == "__main__":
    backtest_v44_thresholds()
