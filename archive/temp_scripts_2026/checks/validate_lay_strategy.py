"""
Lay Strategy Validation Script
==============================
Methods:
1. Monte Carlo Simulation - Shuffle bet order to test consistency
2. K-Fold Cross-Validation - Time-based folds to test generalization  
3. Walk-Forward Validation - Rolling window to simulate live trading

Strategy: Edge < -0.50, Price < $2.00, Prob < 20%, Target Profit 4% Staking
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "C:/Users/Winxy/Documents/Greyhound racing/greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_prod.json"

COMMISSION = 0.08
BANKROLL = 200.0
MAX_PRICE = 2.00
MAX_PROB = 0.20
MAX_EDGE = -0.50

# ------------------------------------------------------------------------------
# DATA LOADING & PREDICTION
# ------------------------------------------------------------------------------
def load_and_predict():
    """Load data, engineer features, and predict probabilities"""
    print("[1/3] Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Weight,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.FinishTime > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Preprocessing
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    df['StartPrice'] = pd.to_numeric(df['StartPrice'], errors='coerce').fillna(0)
    df = df[df['StartPrice'] > 1.0].copy()
    
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(0)
    
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    
    # Feature Engineering
    print("[2/3] Engineering Features...")
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)
            
    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['Margin_avg'] = df.groupby('GreyhoundID')['Margin1'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime']
    df['RunSpeed_avg'] = df.groupby('GreyhoundID')['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
        
    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Predict
    print("[3/3] Predicting Probabilities...")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dtest = xgb.DMatrix(df[feature_cols])
    df['Prob_V33'] = model.predict(dtest)
    df['Implied_Prob'] = 1 / df['StartPrice']
    # Use ABSOLUTE edge (prob - implied) - matches backtest
    df['Edge'] = df['Prob_V33'] - df['Implied_Prob']
    
    # Filter for Lay Opportunities
    mask = (df['StartPrice'] <= MAX_PRICE) & \
           (df['Prob_V33'] < MAX_PROB) & \
           (df['Edge'] < MAX_EDGE)
    
    bets = df[mask].copy()
    bets = bets.sort_values('date_dt').reset_index(drop=True)
    
    print(f"Found {len(bets)} Lay Opportunities")
    return bets

def calculate_pl(bets, bankroll=BANKROLL):
    """Calculate P/L for a set of bets using Target Profit 4% staking"""
    # Target Profit 4% of Bank
    target = bankroll * 0.04
    bets = bets.copy()
    bets['Lay_Stake'] = target / (1 - COMMISSION)
    bets['Risk'] = bets['Lay_Stake'] * (bets['StartPrice'] - 1)
    
    bets['Lay_Result'] = np.where(bets['Place'] > 1, 'WIN', 'LOSS')
    
    bets['PL'] = np.where(
        bets['Lay_Result'] == 'WIN',
        bets['Lay_Stake'] * (1 - COMMISSION),
        -bets['Risk']
    )
    
    return bets

def get_metrics(bets):
    """Calculate performance metrics"""
    if len(bets) == 0:
        return {'profit': 0, 'roi': 0, 'sr': 0, 'dd': 0, 'n_bets': 0}
    
    total_profit = bets['PL'].sum()
    total_risk = bets['Risk'].sum()
    roi = (total_profit / total_risk) * 100 if total_risk > 0 else 0
    
    wins = len(bets[bets['Lay_Result'] == 'WIN'])
    sr = (wins / len(bets)) * 100
    
    # Drawdown
    bets = bets.copy()
    bets['Bank'] = BANKROLL + bets['PL'].cumsum()
    peak = bets['Bank'].cummax()
    dd = ((peak - bets['Bank']) / peak).max() * 100
    
    return {
        'profit': total_profit,
        'roi': roi,
        'sr': sr,
        'dd': dd,
        'n_bets': len(bets)
    }

# ------------------------------------------------------------------------------
# 1. MONTE CARLO SIMULATION
# ------------------------------------------------------------------------------
def monte_carlo_validation(bets, n_simulations=1000):
    """
    Shuffle bet order N times to test if results are sequence-dependent
    """
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION")
    print("="*80)
    print(f"Running {n_simulations} simulations...")
    
    bets = calculate_pl(bets)
    base_metrics = get_metrics(bets)
    
    profits = []
    rois = []
    
    for i in range(n_simulations):
        shuffled = bets.sample(frac=1, random_state=i).reset_index(drop=True)
        shuffled = calculate_pl(shuffled)
        metrics = get_metrics(shuffled)
        profits.append(metrics['profit'])
        rois.append(metrics['roi'])
        
        if (i + 1) % 200 == 0:
            print(f"  Completed {i+1}/{n_simulations}...")
    
    profits = np.array(profits)
    rois = np.array(rois)
    
    # Statistics
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"Base Profit: ${base_metrics['profit']:.2f}")
    print(f"Mean Profit: ${profits.mean():.2f} (Std: ${profits.std():.2f})")
    print(f"5th Percentile: ${np.percentile(profits, 5):.2f}")
    print(f"95th Percentile: ${np.percentile(profits, 95):.2f}")
    print(f"% Profitable Runs: {(profits > 0).mean() * 100:.1f}%")
    print(f"Mean ROI: {rois.mean():.2f}% (Std: {rois.std():.2f}%)")
    print("-"*60)
    
    return {'profits': profits, 'rois': rois, 'base': base_metrics}

# ------------------------------------------------------------------------------
# 2. K-FOLD CROSS-VALIDATION (Time-Based)
# ------------------------------------------------------------------------------
def kfold_validation(bets, k=5):
    """
    Split data into K time-based folds and test each
    """
    print("\n" + "="*80)
    print(f"K-FOLD CROSS-VALIDATION (K={k})")
    print("="*80)
    
    # Sort by date
    bets = bets.sort_values('date_dt').reset_index(drop=True)
    
    # Split into K folds
    fold_size = len(bets) // k
    results = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(bets)
        
        fold = bets.iloc[start_idx:end_idx].copy()
        fold = calculate_pl(fold)
        metrics = get_metrics(fold)
        
        date_range = f"{fold['date_dt'].min().strftime('%Y-%m-%d')} to {fold['date_dt'].max().strftime('%Y-%m-%d')}"
        
        print(f"Fold {i+1}: {metrics['n_bets']:>4} bets | Profit: ${metrics['profit']:>8.2f} | ROI: {metrics['roi']:>6.2f}% | SR: {metrics['sr']:>5.1f}% | {date_range}")
        
        results.append(metrics)
    
    # Summary
    profits = [r['profit'] for r in results]
    rois = [r['roi'] for r in results]
    
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"Mean Profit per Fold: ${np.mean(profits):.2f} (Std: ${np.std(profits):.2f})")
    print(f"Mean ROI per Fold: {np.mean(rois):.2f}% (Std: {np.std(rois):.2f}%)")
    print(f"Folds Profitable: {sum(1 for p in profits if p > 0)}/{k}")
    print("-"*60)
    
    return results

# ------------------------------------------------------------------------------
# 3. WALK-FORWARD VALIDATION
# ------------------------------------------------------------------------------
def walkforward_validation(bets, train_months=6, test_months=1):
    """
    Simulate live trading: Train on N months, test on next month, roll forward
    """
    print("\n" + "="*80)
    print(f"WALK-FORWARD VALIDATION (Train: {train_months}mo, Test: {test_months}mo)")
    print("="*80)
    
    bets = bets.sort_values('date_dt').reset_index(drop=True)
    
    # Get date range
    min_date = bets['date_dt'].min()
    max_date = bets['date_dt'].max()
    
    results = []
    current_start = min_date
    
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > max_date:
            break
        
        # Test period bets (we're using the same model, just time-splitting the data)
        test_bets = bets[(bets['date_dt'] >= train_end) & (bets['date_dt'] < test_end)].copy()
        
        if len(test_bets) == 0:
            current_start += pd.DateOffset(months=test_months)
            continue
        
        test_bets = calculate_pl(test_bets)
        metrics = get_metrics(test_bets)
        
        test_period = f"{train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"
        print(f"Test: {test_period} | {metrics['n_bets']:>3} bets | Profit: ${metrics['profit']:>7.2f} | ROI: {metrics['roi']:>6.2f}%")
        
        results.append({
            'period': test_period,
            **metrics
        })
        
        current_start += pd.DateOffset(months=test_months)
    
    # Summary
    if results:
        profits = [r['profit'] for r in results]
        rois = [r['roi'] for r in results]
        
        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        print(f"Total Periods: {len(results)}")
        print(f"Total Profit: ${sum(profits):.2f}")
        print(f"Mean Monthly Profit: ${np.mean(profits):.2f} (Std: ${np.std(profits):.2f})")
        print(f"Mean Monthly ROI: {np.mean(rois):.2f}% (Std: {np.std(rois):.2f}%)")
        print(f"Profitable Months: {sum(1 for p in profits if p > 0)}/{len(results)} ({sum(1 for p in profits if p > 0)/len(results)*100:.0f}%)")
        print("-"*60)
    
    return results

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("LAY STRATEGY VALIDATION SUITE")
    print("="*80)
    print(f"Strategy: Edge < {MAX_EDGE}, Price < ${MAX_PRICE}, Prob < {MAX_PROB*100}%")
    print(f"Staking: Target Profit 4% of Bank (${BANKROLL})")
    print("="*80)
    
    # Load data
    bets = load_and_predict()
    
    if len(bets) == 0:
        print("No bets found matching criteria!")
        exit()
    
    # Run validations
    mc_results = monte_carlo_validation(bets, n_simulations=1000)
    kf_results = kfold_validation(bets, k=5)
    wf_results = walkforward_validation(bets, train_months=6, test_months=1)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
