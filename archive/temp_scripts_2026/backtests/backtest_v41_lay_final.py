
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

def backtest_lay_v41():
    print("="*60)
    print("BACKTEST: V41/V43 LAY STRATEGY (Drifter > 0.65)")
    print("="*60)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Fetch Data (2024-2025)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, ge.LTP,
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
    AND ge.Price5Min IS NOT NULL AND ge.Price5Min > 0
    AND ge.Price5Min <= 40.0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Data Loaded: {len(df)} rows")
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    # Load Models
    model_v41 = joblib.load("models/xgb_v41_final.pkl")   # Back Prob
    model_v43 = joblib.load("models/xgb_v43_drifter.pkl") # Drift Prob
    
    # Prepare Features
    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    def safe_predict(model, data_df, cols=None):
        if cols: data_df = data_df[cols]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(data_df)[:, 1]
        else:
            dmat = xgb.DMatrix(data_df)
            return model.predict(dmat)

    # 1. Base Probability
    df['V41_Prob'] = safe_predict(model_v41, df, v41_cols)
    df['V41_Price'] = 1.0 / df['V41_Prob']
    
    # 2. Alpha Features for Drifter Model
    df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
    df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    # 3. Drift Probability
    df['Drift_Prob'] = safe_predict(model_v43, df, alpha_feats)
    
    # 4. Strategy: LAY if Drift_Prob >= 0.65
    threshold = 0.65
    ts = [0.65] 
    
    print(f"\nTesting Lay Threshold: {threshold}")
    print(f"{'Thresh':<6} | {'Bets':<6} | {'Win% (Lay)':<10} | {'Liability':<10} | {'Profit':<10} | {'ROI':<6}")
    print("-" * 70)
    
    for th in ts:
        # Selection logic
        potential_bets = df[df['Drift_Prob'] >= th].copy()
        
        # EXCLUSION RULE: Exclude races where we would lay > 2 runners
        race_counts = potential_bets['RaceID'].value_counts()
        exclude_races = race_counts[race_counts > 2].index
        
        bets = potential_bets[~potential_bets['RaceID'].isin(exclude_races)].copy()
        
        excluded_count = len(potential_bets) - len(bets)
        if excluded_count > 0:
            print(f"   [Rule] Excluded {len(exclude_races)} races ({excluded_count} bets) with > 2 lays.")
        
        if len(bets) == 0:
            print(f"{th:<6} | 0      | 0.0%       | $0         | $0.00      | 0.0%")
            continue
            
        stake = 10.0
        
        # Lay Bet Logic:
        # We win if Position != 1
        bets['Lay_Win'] = (bets['Position'] != 1).astype(int)
        
        # PnL Calculation
        # If Lay Win (Dog Loses): Profit = Stake * (1 - Comm)
        # If Lay Loss (Dog Wins): Loss = Stake * (Price - 1)
        # Using Price5Min as execution price (or LTP if better?) -> User said "Live prices", usually Price5Min is the trigger.
        # Let's use Price5Min for liability calculation.
        
        comm = 0.05 # 5% commission on net winnings per market, simplified here to per bet
        
        bets['Liability'] = stake * (bets['Price5Min'] - 1)
        
        bets['PnL'] = np.where(
            bets['Lay_Win'] == 1,
            stake * (1 - comm),       # Won the lay (kept stake - comm)
            -1 * bets['Liability']    # Lost the lay (pay out liability)
        )
        
        total_bets = len(bets)
        win_rate = (bets['Lay_Win'].sum() / total_bets) * 100
        total_pnl = bets['PnL'].sum()
        total_risk = bets['Liability'].sum() # ROI on Liability? Or Stake? Usually Liability for Lay.
        
        # ROI on Turnover (Liability Risked)
        roi = (total_pnl / total_risk) * 100 if total_risk > 0 else 0
        
        print(f"{th:<6.2f} | {total_bets:<6} | {win_rate:<10.1f}% | ${total_risk:<10,.0f} | ${total_pnl:<10,.2f} | {roi:<6.1f}%")
        
        # Check by Price Band
        print("\n   By Price Band (< $40):")
        labels = ['< $3', '$3-$6', '$6-$10', '$10-$20', '$20+']
        bins = [0, 3, 6, 10, 20, 41]
        bets['PriceBin'] = pd.cut(bets['Price5Min'], bins=bins, labels=labels)
        
        summary = bets.groupby('PriceBin')['PnL'].sum()
        counts = bets.groupby('PriceBin')['PnL'].count()
        
        for lab in labels:
            c = counts.get(lab, 0)
            p = summary.get(lab, 0)
            print(f"   {lab:<8}: {c:>5} bets | ${p:>8.2f}")

if __name__ == "__main__":
    backtest_lay_v41()
