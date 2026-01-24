"""
BACKTEST: Cover Strategy V3 - Sensitivity Analysis
- Condition: Races with >= 2 Lay Bets.
- Action: BACK the highest confidence steamer.
- Variable: Steam Prob Threshold (0.40, 0.41, 0.42, 0.43, 0.44, 0.45)
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def run_backtest():
    print("="*80)
    print("BACKTEST: COVER STRATEGY SENSITIVITY (40% - 45%)")
    print("Condition: Back Highest Steam_Prob (Price < $10)")
    print("="*80)
    
    # 1. LOAD DATA
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        ge.Price5Min,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.GreyhoundName as Dog, g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2023-06-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 2. FEATURES
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['MeetingDate'] <= '2025-12-31') & 
        (df['Price5Min'] > 0)
    ].sort_values('MeetingDate').copy()
    
    print(f"Population: {len(df_bt)} runs")

    # 3. MODELS
    model_v41 = joblib.load(MODEL_V41)
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)

    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df_bt.columns: df_bt[c] = 0
        df_bt[c] = pd.to_numeric(df_bt[c], errors='coerce').fillna(0)
        
    dmatrix = xgb.DMatrix(df_bt[v41_cols])
    df_bt['V41_Prob'] = model_v41.predict(dmatrix)
    df_bt['V41_Price'] = 1.0 / df_bt['V41_Prob']
    
    df_bt['Discrepancy'] = df_bt['Price5Min'] / df_bt['V41_Price']
    df_bt['Price_Diff'] = df_bt['Price5Min'] - df_bt['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    # 4. SENSITIVITY ANALYSIS
    thresholds = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    results = []

    # Pre-calculate Lays once (expensive)
    print("Identifying races with 2+ Lays...")
    
    # Vectorized Lay Identification
    # Price < 4 -> 0.55
    # Price < 8 -> 0.60
    # Price >= 8 -> 0.65
    df_bt['IsLay'] = False
    
    cond_low = (df_bt['Price5Min'] < 4.0) & (df_bt['Drift_Prob'] >= 0.55)
    cond_mid = (df_bt['Price5Min'] >= 4.0) & (df_bt['Price5Min'] < 8.0) & (df_bt['Drift_Prob'] >= 0.60)
    cond_high = (df_bt['Price5Min'] >= 8.0) & (df_bt['Drift_Prob'] >= 0.65)
    
    df_bt.loc[cond_low | cond_mid | cond_high, 'IsLay'] = True
    
    # Find Races
    lay_counts = df_bt[df_bt['IsLay']].groupby('RaceID').size()
    target_races = lay_counts[lay_counts >= 2].index
    
    print(f"Found {len(target_races)} races with 2+ Lays.")
    
    # Subset Data to Target Races Only (Speed Optimization)
    target_df = df_bt[df_bt['RaceID'].isin(target_races)].copy()
    grouped_target = target_df.groupby('RaceID')

    for thresh in thresholds:
        print(f"Testing Threshold > {thresh:.2f}...")
        
        stats = {
            'bets': 0, 'wins': 0, 'pnl': 0.0, 'invested': 0.0
        }
        curr_bank = 2000.0
        
        for race_id, race_df in grouped_target:
            # Select Cover Bet
            back_candidates = race_df[
                (race_df['Price5Min'] < 10.0) & 
                (race_df['Steam_Prob'] > thresh)
            ]
            
            if not back_candidates.empty:
                # Best only
                best = back_candidates.sort_values('Steam_Prob', ascending=False).iloc[0]
                
                # Exec
                price = best['Price5Min']
                
                # Stake
                # 4% Target
                bank_roll = curr_bank if curr_bank > 100 else 200.0
                target_profit = bank_roll * 0.04
                stake = target_profit / (price - 1.0) if price > 1.0 else 0
                
                # Result
                win = (best['Position'] == 1)
                pnl = abs(stake * (price - 1) * 0.95) if win else -stake
                if not win: pnl = -stake # Explicit negative
                
                stats['bets'] += 1
                if win: stats['wins'] += 1
                stats['pnl'] += pnl
                stats['invested'] += stake
                curr_bank += pnl
        
        roi = (stats['pnl'] / stats['invested'] * 100) if stats['invested'] > 0 else 0.0
        win_rate = (stats['wins'] / stats['bets'] * 100) if stats['bets'] > 0 else 0.0
        
        results.append({
            'Threshold': thresh,
            'Bets': stats['bets'],
            'WinRate': win_rate,
            'PnL': stats['pnl'],
            'ROI': roi
        })

    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*60)
    print(f"{'Prob >':<10} | {'Bets':<6} | {'Win %':<8} | {'PnL ($)':<10} | {'ROI %':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['Threshold']:<10.2f} | {r['Bets']:<6} | {r['WinRate']:<8.1f} | {r['PnL']:<10.2f} | {r['ROI']:<8.2f}")
    
    print("-" * 60)
    
    # CSV
    pd.DataFrame(results).to_csv("backtest_cover_sensitivity.csv", index=False)
    print("Saved to backtest_cover_sensitivity.csv")

if __name__ == "__main__":
    run_backtest()
