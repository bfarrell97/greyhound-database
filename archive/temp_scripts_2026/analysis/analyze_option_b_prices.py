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

def analyze_prices():
    print("Loading Data (2024-2025) for Price Analysis...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, ge.Price30Min,
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
    ORDER BY rm.MeetingDate ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    
    # Predict
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df_clean[features])
    df_clean['Prob'] = model.predict(dtest)
    df_clean['ImpliedProb'] = 1.0 / df_clean['BSP']
    df_clean['Edge'] = df_clean['Prob'] - df_clean['ImpliedProb']
    
    # Filter: Option B
    # Edge > 0.21, Prob > 0.28, BSP < 7.90
    bets = df_clean[
        (df_clean['BSP'] >= 1.50) & 
        (df_clean['BSP'] <= 7.90) &
        (df_clean['Edge'] > 0.21) &
        (df_clean['Prob'] > 0.28) &
        (df_clean['Price5Min'].notnull()) & 
        (df_clean['Price5Min'] > 1.0)
    ].copy()
    
    if bets.empty:
        print("No bets found matching Option B filters with valid 5Min Prices.")
        return

    bets['Win'] = (bets['win'] == 1).astype(int)
    stake = 10
    
    # Calculate Returns
    bets['Return_BSP'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
    bets['Return_5Min'] = np.where(bets['Win'] == 1, stake * (bets['Price5Min'] - 1) * 0.92 + stake, 0)
    
    profit_bsp = bets['Return_BSP'].sum() - (len(bets) * stake)
    roi_bsp = (profit_bsp / (len(bets) * stake)) * 100
    
    profit_5min = bets['Return_5Min'].sum() - (len(bets) * stake)
    roi_5min = (profit_5min / (len(bets) * stake)) * 100
    
    # Compare
    bets['PriceDiff'] = bets['BSP'] - bets['Price5Min']
    drifters = bets[bets['PriceDiff'] > 0]
    steamers = bets[bets['PriceDiff'] < 0]
    stable = bets[bets['PriceDiff'] == 0]
    
    print("\n" + "="*80)
    print("OPTION B: BSP vs 5-MIN FIXED ODDS ANALYSIS")
    print(f"Total Bets: {len(bets)}")
    print("="*80)
    
    print(f"{'Price Source':<15} | {'Profit':<10} | {'ROI':<8}")
    print("-" * 40)
    print(f"{'BSP (Market)':<15} | ${profit_bsp:>9.2f} | {roi_bsp:>7.1f}%")
    print(f"{'Fixed 5-Min':<15} | ${profit_5min:>9.2f} | {roi_5min:>7.1f}%")
    print("-" * 40)
    
    diff = roi_5min - roi_bsp
    better = "5-MIN" if diff > 0 else "BSP"
    print(f"\nWinner: {better} (Diff: {diff:+.1f}%)")
    
    print("\n--- Market Dynamics ---")
    print(f"Steamers (BSP < 5Min): {len(steamers)} ({len(steamers)/len(bets)*100:.1f}%)")
    print(f"Drifters (BSP > 5Min): {len(drifters)} ({len(drifters)/len(bets)*100:.1f}%)")
    print(f"Stable   (BSP = 5Min): {len(stable)} ({len(stable)/len(bets)*100:.1f}%)")
    
    if not steamers.empty:
         s_prof_bsp = steamers['Return_BSP'].sum() - (len(steamers)*stake)
         s_prof_5min = steamers['Return_5Min'].sum() - (len(steamers)*stake)
         print(f"\nSteamer Performance (Taking 5Min would have gained ${s_prof_5min - s_prof_bsp:.0f}):")
         print(f"  BSP Profit: ${s_prof_bsp:.0f} | 5Min Profit: ${s_prof_5min:.0f}")

    if not drifters.empty:
         d_prof_bsp = drifters['Return_BSP'].sum() - (len(drifters)*stake)
         d_prof_5min = drifters['Return_5Min'].sum() - (len(drifters)*stake)
         print(f"\nDrifter Performance (Taking 5Min would have lost ${d_prof_bsp - d_prof_5min:.0f}):")
         print(f"  BSP Profit: ${d_prof_bsp:.0f} | 5Min Profit: ${d_prof_5min:.0f}")

if __name__ == "__main__":
    analyze_prices()
