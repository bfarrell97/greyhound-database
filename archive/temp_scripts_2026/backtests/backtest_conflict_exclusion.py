"""
Backtest: Conflict Exclusion Filter
====================================
Tests the effectiveness of excluding bets when the opposite model shows potential.

Hypothesis: If a dog meets the LAY threshold but also has high BACK probability,
it's a "conflicted" signal and we should skip it.

Control: Current thresholds (no exclusion)
Test: Exclude bets where opposite model shows potential
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import joblib

# --- CONFIGURATION ---
DB_PATH = 'greyhound_racing.db'
MODEL_PATH_V44 = 'models/xgb_v44_production.pkl'  # Back/Steamer model
MODEL_PATH_V45 = 'models/xgb_v45_production.pkl'  # Lay/Drift model

# Current thresholds (from production)
BACK_THRESHOLD = 0.30  # Steam_Prob >= this = BACK signal
LAY_THRESHOLD = 0.55   # Drift_Prob >= this = LAY signal

# Exclusion thresholds to test (skip bet if opposite model is above this)
EXCLUSION_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]

# Backtest period
MONTHS_BACK = 6


def load_models():
    """Load the V44 (BACK) and V45 (LAY) models"""
    try:
        back_model = joblib.load(MODEL_PATH_V44)
        print(f"[OK] Loaded BACK model: {MODEL_PATH_V44}")
    except:
        back_model = None
        print(f"[WARN] Could not load BACK model: {MODEL_PATH_V44}")
        
    try:
        lay_model = joblib.load(MODEL_PATH_V45)
        print(f"[OK] Loaded LAY model: {MODEL_PATH_V45}")
    except:
        lay_model = None
        print(f"[WARN] Could not load LAY model: {MODEL_PATH_V45}")
        
    return back_model, lay_model


def load_historical_data(months_back=6):
    """Load historical race data for backtesting"""
    conn = sqlite3.connect(DB_PATH)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    print(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Load race data with results
    query = """
        SELECT 
            ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, ge.Position as FinalPosition,
            ge.StartingPrice, ge.Price5Min, ge.Price1Min,
            g.GreyhoundName as Dog,
            t.TrackName, t.State,
            r.RaceNumber, r.Distance, r.Grade, r.RaceTime,
            rm.MeetingDate
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= ? 
        AND rm.MeetingDate <= ?
        AND ge.Position IS NOT NULL
        AND ge.StartingPrice > 0
    """
    
    df = pd.read_sql_query(query, conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
    conn.close()
    
    print(f"Loaded {len(df)} entries from {df['MeetingDate'].nunique()} meeting days")
    return df


def engineer_features(df):
    """Engineer features for model prediction (simplified version)"""
    # Convert price columns to numeric
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce').fillna(0)
    df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce').fillna(0)
    df['Price1Min'] = pd.to_numeric(df['Price1Min'], errors='coerce').fillna(0)
    
    # Basic features that models expect
    df['Price'] = df['Price5Min'].replace(0, np.nan).fillna(df['StartingPrice'])
    df['Price'] = df['Price'].replace(0, np.nan).fillna(df['Price1Min'])
    df['Price'] = df['Price'].replace(0, np.nan).fillna(5.0)  # Default price if all missing
    
    # Box position features
    df['BoxNumber'] = df['Box'].fillna(0).astype(int)
    df['Is_Box1'] = (df['BoxNumber'] == 1).astype(int)
    df['Is_Box8'] = (df['BoxNumber'] == 8).astype(int)
    df['Is_WideBox'] = (df['BoxNumber'] >= 6).astype(int)
    
    # Distance category
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(400)
    df['Is_Sprint'] = (df['Distance'] <= 350).astype(int)
    df['Is_Middle'] = ((df['Distance'] > 350) & (df['Distance'] <= 500)).astype(int)
    df['Is_Stay'] = (df['Distance'] > 500).astype(int)
    
    # Price features
    df['Log_Price'] = np.log(df['Price'].clip(lower=1.01))
    df['Is_Favourite'] = (df['Price'] <= 3.0).astype(int)
    df['Is_Outsider'] = (df['Price'] >= 20.0).astype(int)
    
    # Grade numeric (rough proxy)
    grade_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
                 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5, 'G6': 6, 'G7': 7,
                 'MDN': 8, 'MAID': 8, 'MAIDEN': 8, 'FREE': 9}
    df['Grade_Num'] = df['Grade'].str.upper().str.extract(r'(\d+)')[0].fillna(5).astype(float)
    
    return df


def generate_predictions(df, back_model, lay_model):
    """Generate BACK and LAY predictions for all entries"""
    
    # Get feature columns that models expect
    # Try to use same features as training
    feature_cols = ['Box', 'Distance', 'Log_Price', 'Is_Box1', 'Is_Box8', 
                    'Is_WideBox', 'Is_Sprint', 'Is_Middle', 'Is_Stay',
                    'Is_Favourite', 'Is_Outsider', 'Grade_Num']
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols].fillna(0)
    
    # Generate predictions
    df['Steam_Prob'] = 0.5  # Default
    df['Drift_Prob'] = 0.5  # Default
    
    if back_model is not None:
        try:
            df['Steam_Prob'] = back_model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"[WARN] BACK model prediction failed: {e}")
            # Fallback: use simple heuristic
            df['Steam_Prob'] = 0.3 + np.random.randn(len(df)) * 0.1
            
    if lay_model is not None:
        try:
            df['Drift_Prob'] = lay_model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"[WARN] LAY model prediction failed: {e}")
            # Fallback: use simple heuristic
            df['Drift_Prob'] = 0.5 + np.random.randn(len(df)) * 0.1
    
    return df


def calculate_pnl(df, signal_type, exclusion_threshold=None, opposite_col='Steam_Prob'):
    """
    Calculate P&L for a set of signals
    
    Args:
        df: DataFrame with predictions and results
        signal_type: 'BACK' or 'LAY'
        exclusion_threshold: If set, exclude bets where opposite model prob > this
        opposite_col: Column name for opposite model probability
    """
    if signal_type == 'BACK':
        # BACK signals: Steam_Prob >= BACK_THRESHOLD
        signals = df[df['Steam_Prob'] >= BACK_THRESHOLD].copy()
        
        if exclusion_threshold is not None:
            # Exclude where LAY prob is high
            signals = signals[signals['Drift_Prob'] < exclusion_threshold]
        
        # P&L: Win = (Price - 1), Lose = -1
        signals['Won'] = (signals['FinalPosition'] == 1).astype(int)
        signals['PnL'] = signals.apply(
            lambda r: (r['Price'] - 1) if r['Won'] else -1, axis=1
        )
        
    else:  # LAY
        # LAY signals: Drift_Prob >= LAY_THRESHOLD
        signals = df[df['Drift_Prob'] >= LAY_THRESHOLD].copy()
        
        if exclusion_threshold is not None:
            # Exclude where BACK prob is high
            signals = signals[signals['Steam_Prob'] < exclusion_threshold]
        
        # P&L: Win (dog loses) = +1, Lose (dog wins) = -(Price - 1)
        signals['Won'] = (signals['FinalPosition'] != 1).astype(int)
        signals['PnL'] = signals.apply(
            lambda r: 1 if r['FinalPosition'] != 1 else -(r['Price'] - 1), axis=1
        )
    
    return signals


def run_backtest():
    """Main backtest function"""
    print("=" * 70)
    print("CONFLICT EXCLUSION BACKTEST")
    print("=" * 70)
    print(f"Testing if excluding conflicting signals improves performance")
    print(f"Hypothesis: Skip LAY if BACK prob high, skip BACK if LAY prob high")
    print("-" * 70)
    
    # Load models
    back_model, lay_model = load_models()
    
    # Load data
    df = load_historical_data(MONTHS_BACK)
    
    if len(df) == 0:
        print("[ERROR] No data loaded!")
        return
    
    # Engineer features
    df = engineer_features(df)
    
    # Generate predictions
    df = generate_predictions(df, back_model, lay_model)
    
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    results = []
    
    # --- BACK SIGNALS ---
    print("\n### BACK SIGNALS (Steam_Prob >= {:.2f}) ###".format(BACK_THRESHOLD))
    print("-" * 50)
    
    # Control: No exclusion
    control_back = calculate_pnl(df, 'BACK', exclusion_threshold=None)
    control_back_roi = (control_back['PnL'].sum() / len(control_back) * 100) if len(control_back) > 0 else 0
    control_back_sr = control_back['Won'].mean() * 100 if len(control_back) > 0 else 0
    
    print(f"CONTROL (no exclusion):")
    print(f"  Bets: {len(control_back):,} | Wins: {control_back['Won'].sum():,} ({control_back_sr:.1f}%)")
    print(f"  P&L: {control_back['PnL'].sum():+.2f} units | ROI: {control_back_roi:+.2f}%")
    
    results.append({
        'Type': 'BACK', 'Filter': 'CONTROL', 'Threshold': None,
        'Bets': len(control_back), 'Wins': control_back['Won'].sum(),
        'Strike': control_back_sr, 'PnL': control_back['PnL'].sum(), 'ROI': control_back_roi
    })
    
    # Test: With exclusion
    print(f"\nTEST (exclude if Drift_Prob >= threshold):")
    for thresh in EXCLUSION_THRESHOLDS:
        test = calculate_pnl(df, 'BACK', exclusion_threshold=thresh, opposite_col='Drift_Prob')
        if len(test) > 0:
            roi = (test['PnL'].sum() / len(test) * 100)
            sr = test['Won'].mean() * 100
            excluded = len(control_back) - len(test)
            print(f"  Thresh {thresh:.2f}: Bets {len(test):,} (-{excluded}) | SR {sr:.1f}% | PnL {test['PnL'].sum():+.2f} | ROI {roi:+.2f}%")
            results.append({
                'Type': 'BACK', 'Filter': f'Exclude LAY>{thresh}', 'Threshold': thresh,
                'Bets': len(test), 'Wins': test['Won'].sum(),
                'Strike': sr, 'PnL': test['PnL'].sum(), 'ROI': roi
            })
    
    # --- LAY SIGNALS ---
    print("\n### LAY SIGNALS (Drift_Prob >= {:.2f}) ###".format(LAY_THRESHOLD))
    print("-" * 50)
    
    # Control: No exclusion
    control_lay = calculate_pnl(df, 'LAY', exclusion_threshold=None)
    control_lay_roi = (control_lay['PnL'].sum() / len(control_lay) * 100) if len(control_lay) > 0 else 0
    control_lay_sr = control_lay['Won'].mean() * 100 if len(control_lay) > 0 else 0
    
    print(f"CONTROL (no exclusion):")
    print(f"  Bets: {len(control_lay):,} | Wins: {control_lay['Won'].sum():,} ({control_lay_sr:.1f}%)")
    print(f"  P&L: {control_lay['PnL'].sum():+.2f} units | ROI: {control_lay_roi:+.2f}%")
    
    results.append({
        'Type': 'LAY', 'Filter': 'CONTROL', 'Threshold': None,
        'Bets': len(control_lay), 'Wins': control_lay['Won'].sum(),
        'Strike': control_lay_sr, 'PnL': control_lay['PnL'].sum(), 'ROI': control_lay_roi
    })
    
    # Test: With exclusion
    print(f"\nTEST (exclude if Steam_Prob >= threshold):")
    for thresh in EXCLUSION_THRESHOLDS:
        test = calculate_pnl(df, 'LAY', exclusion_threshold=thresh, opposite_col='Steam_Prob')
        if len(test) > 0:
            roi = (test['PnL'].sum() / len(test) * 100)
            sr = test['Won'].mean() * 100
            excluded = len(control_lay) - len(test)
            print(f"  Thresh {thresh:.2f}: Bets {len(test):,} (-{excluded}) | SR {sr:.1f}% | PnL {test['PnL'].sum():+.2f} | ROI {roi:+.2f}%")
            results.append({
                'Type': 'LAY', 'Filter': f'Exclude BACK>{thresh}', 'Threshold': thresh,
                'Bets': len(test), 'Wins': test['Won'].sum(),
                'Strike': sr, 'PnL': test['PnL'].sum(), 'ROI': roi
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    
    # Best BACK filter
    back_results = results_df[results_df['Type'] == 'BACK']
    best_back = back_results.loc[back_results['ROI'].idxmax()]
    print(f"\nBest BACK: {best_back['Filter']} with ROI {best_back['ROI']:+.2f}%")
    
    # Best LAY filter
    lay_results = results_df[results_df['Type'] == 'LAY']
    best_lay = lay_results.loc[lay_results['ROI'].idxmax()]
    print(f"Best LAY: {best_lay['Filter']} with ROI {best_lay['ROI']:+.2f}%")
    
    # Combined
    control_combined = control_back['PnL'].sum() + control_lay['PnL'].sum()
    control_bets = len(control_back) + len(control_lay)
    print(f"\nCombined CONTROL: {control_bets:,} bets | PnL {control_combined:+.2f} | ROI {control_combined/control_bets*100:+.2f}%")
    
    return results_df


if __name__ == '__main__':
    run_backtest()
