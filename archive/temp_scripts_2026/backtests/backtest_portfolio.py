import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def backtest_portfolio():
    print("Loading Data (2024-2025) for Portfolio Backtest...")
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
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Feature Engineering
    print("Engineering Features...")
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    df_clean = df_clean.sort_values(['MeetingDate', 'RaceID']) # Time order
    
    # Lag Features (Steam & Drift)
    print("Calculating Rolling Lags...")
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer_Hist'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    df_clean['Is_Drifter_Hist'] = (df_clean['MoveRatio'] < 0.95).astype(int)
    
    # Steam Lags
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    ).fillna(0)
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    ).fillna(0)
    
    # Drift Lags
    df_clean['Prev_Drift'] = df_clean.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
    df_clean['Dog_Rolling_Drift_10'] = df_clean.groupby('GreyhoundID')['Prev_Drift'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    ).fillna(0)
    df_clean['Trainer_Prev_Drift'] = df_clean.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
    df_clean['Trainer_Rolling_Drift_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    ).fillna(0)
    
    # 2. Base Model Probabilities
    print("Generating Model Predictions...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    # Using V44 Test (Out-of-Sample)
    print("Loading V44 Test Model (Unbiased)...")
    model_v44 = joblib.load('models/xgb_v44_test.pkl')
    # Using V45 Test (Out-of-Sample)
    model_v45 = joblib.load('models/xgb_v45_test.pkl') 
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
    
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # Feature Lists
    v44_features = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    v45_features = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    
    # Predict
    for c in v44_features: 
        if c not in df_clean.columns: df_clean[c] = 0
    X_v44 = df_clean[v44_features]
    df_clean['Steam_Prob'] = model_v44.predict_proba(X_v44)[:, 1]
    
    for c in v45_features:
        if c not in df_clean.columns: df_clean[c] = 0
    X_v45 = df_clean[v45_features]
    df_clean['Drift_Prob'] = model_v45.predict_proba(X_v45)[:, 1]
    
    print(f"DEBUG: Steam_Prob Mean={df_clean['Steam_Prob'].mean():.4f}, Max={df_clean['Steam_Prob'].max():.4f}")
    print(f"DEBUG: Drift_Prob Mean={df_clean['Drift_Prob'].mean():.4f}, Max={df_clean['Drift_Prob'].max():.4f}")
    
    # Check Counts pre-filter
    c_drift = (df_clean['Drift_Prob'] >= 0.65).sum()
    c_steam = (df_clean['Steam_Prob'] >= 0.40).sum()
    print(f"DEBUG: Dogs > 0.65 Drift (Raw): {c_drift}")
    print(f"DEBUG: Dogs > 0.40 Steam (Raw): {c_steam}")
    
    # Check Prices
    print(f"DEBUG: Price5Min Min={df_clean['Price5Min'].min()}, Max={df_clean['Price5Min'].max()}, Mean={df_clean['Price5Min'].mean()}")
    c_valid_price = (df_clean['Price5Min'] < 30.0).sum()
    print(f"DEBUG: Dogs with Price < 30: {c_valid_price} / {len(df_clean)}")

    # 3. Generating Signals
    print("Generating Signals...")
    df_clean['Signal'] = 'PASS'
    
    # Back Strategy
    mask_back = (df_clean['Steam_Prob'] >= 0.40) & (df_clean['Price5Min'] < 30.0)
    df_clean.loc[mask_back, 'Signal'] = 'BACK'
    print(f"DEBUG: Back Signals Set: {mask_back.sum()}")
    
    # Lay Strategy
    mask_lay = (df_clean['Drift_Prob'] >= 0.65) & (df_clean['Price5Min'] < 30.0)
    df_clean.loc[mask_lay, 'Signal'] = 'LAY' 
    print(f"DEBUG: Lay Signals Set: {mask_lay.sum()}")
    
    # Check overlap (just for info)
    overlap = mask_back & mask_lay
    print(f"DEBUG: Overlap: {overlap.sum()}")
    if overlap.sum() > 0:
        df_clean.loc[overlap, 'Signal'] = 'PASS'

    signals = df_clean[df_clean['Signal'] != 'PASS'].copy()
    print(f"DEBUG: Final Signal Count: {len(signals)}")
    if len(signals) > 0:
        print("DEBUG: Sample Signals:")
        print(signals[['MeetingDate', 'GreyhoundID', 'Signal', 'Price5Min', 'Steam_Prob', 'Drift_Prob']].head())
    
    signals['Dog_Win'] = (pd.to_numeric(signals['Position'], errors='coerce') == 1).astype(int)
    
    # 4. Simulation Loop
    print("\nSimulating Portfolio (Bank $200)...")
    bank = 200.0
    history = []
    
    # Group by Race to simulate sequential betting? 
    # Or just iterate row by row (assuming sequential races).
    # Since we sorted by MeetingDate + RaceID, it's roughly sequential.
    
    active_bets = 0
    wins = 0
    losses = 0
    
    # Staking Config (Hybrid)
    BACK_TARGET_PCT = 0.04 # 4% Target Profit
    LAY_LIABILITY_PCT = 0.025 # 2.5% Liability Limit
    COMMISSION = 0.05
    
    # To properly simulate "Bank Growth", we iterate races.
    # But getting strict time order is tricky across tracks.
    # We'll just iterate the DataFrame as is (Date sorted).
    
    for idx, row in signals.iterrows():
        current_price = row['Price5Min']
        outcome = row['Dog_Win'] # 1 or 0
        sig_type = row['Signal']
        
        stake = 0.0
        pnl = 0.0
        
        if sig_type == 'BACK':
            # Target Profit Staking
            # Stake = (Bank * 0.06) / (Price - 1)
            target = bank * BACK_TARGET_PCT
            stake = target / (current_price - 1) if current_price > 1 else 0
            
            # Sanity check stake
            if stake > bank: stake = bank # All in cap
            
            if outcome == 1:
                revenue = stake * (current_price - 1) * (1 - COMMISSION)
                pnl = revenue
                wins += 1
            else:
                pnl = -stake
                losses += 1
                
        elif sig_type == 'LAY':
            # Liability Staking
            # Liability = Bank * 0.10
            # Stake = Liability / (Price - 1)
            liability_limit = bank * LAY_LIABILITY_PCT
            stake = liability_limit / (current_price - 1) if current_price > 1 else 0
            
            # Lay Stake is the Backer's stake we accept.
            # Our Liability is Stake * (Price - 1).
            actual_liability = stake * (current_price - 1)
            
            # Sanity check
            if actual_liability > bank:
                # Reduce stake to fit bank
                actual_liability = bank
                stake = actual_liability / (current_price - 1)
            
            if outcome == 1:
                # Dog Won -> We Lose Liability
                pnl = -actual_liability
                losses += 1
            else:
                # Dog Lost -> We Win Stake
                revenue = stake * (1 - COMMISSION)
                pnl = revenue
                wins += 1
        
        bank += pnl
        history.append({
            'Date': row['MeetingDate'],
            'Signal': sig_type,
            'Price': current_price,
            'Result': 'WIN' if pnl > 0 else 'LOSS',
            'PnL': pnl,
            'Bank': bank
        })
        
        if bank <= 0:
            print("BANKRUPT!")
            break

    results = pd.DataFrame(history)
    
    # Stats
    total_bets = wins + losses
    final_bank = bank
    roi = ((final_bank - 200) / 200) * 100
    
    print("\n" + "="*60)
    print("PORTFOLIO BACKTEST RESULTS (2024-2025)")
    print("Strategy: V44 Back (>0.40) + V45 Lay (>0.65)")
    print("Staking: Back (Target 4%), Lay (Liab 2.5%)")
    print("-" * 60)
    print(f"Start Bank:      $200.00")
    print(f"Final Bank:      ${final_bank:.2f}")
    print(f"Net Profit:      ${final_bank - 200:.2f}")
    print(f"ROI (Bank):      {roi:+.2f}%")
    print(f"Total Bets:      {total_bets}")
    print(f"Win Rate:        {wins/total_bets*100:.1f}% ({wins} W / {losses} L)")
    
    # Drawdown
    if len(results) > 0:
        results['Peak'] = results['Bank'].cummax()
        results['Drawdown'] = (results['Bank'] - results['Peak']) / results['Peak']
        max_dd = results['Drawdown'].min()
        print(f"Max Drawdown:    {max_dd*100:.2f}%")
    else:
        print("No bets placed.")

    print("="*60)
    print("Note: V44 predictions may act 'optimistically' as 2024 data is in training set.")

if __name__ == "__main__":
    backtest_portfolio()
