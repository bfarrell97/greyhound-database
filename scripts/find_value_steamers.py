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

def find_value_steamers():
    print("Loading Data (2025) for Value Steamer Discovery...")
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
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.Price5Min IS NOT NULL
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
    
    # Define "Value Steamer"
    # 1. Model sees value (Positive Edge)
    # 2. Market steams it (BSP < Early Price)
    # 3. Profitable Early Price (> 1.05 ROI potentially)
    
    # Let's pivot by EDGE bins first
    df_clean['Steamed'] = (df_clean['BSP'] < df_clean['Price5Min']).astype(int)
    
    print("\n" + "="*80)
    print("VALUE STEAMER SCAN (Where does Edge meet Steam?)")
    print("="*80)
    
    edge_bins = [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]
    labels = ['Very Neg', 'Neg', 'Neutral', 'Pos', 'High', 'Very High']
    df_clean['EdgeBin'] = pd.cut(df_clean['Edge'], bins=edge_bins, labels=labels)
    
    print(f"{'Edge Bin':<12} | {'Bets':<6} | {'Steam%':<6} | {'Early ROI':<10} | {'BSP ROI':<8}")
    print("-" * 60)
    
    for label in labels:
        subset = df_clean[df_clean['EdgeBin'] == label].copy()
        if len(subset) < 100: continue
            
        count = len(subset)
        steam_rate = subset['Steamed'].mean() * 100
        
        # Win Rate
        subset['Win'] = (subset['win']==1).astype(int)
        
        # Early ROI
        ret_early = np.where(subset['Win']==1, 10*(subset['Price5Min']-1)*0.92, -10).sum()
        roi_early = ret_early / (count*10) * 100
        
        # BSP ROI
        ret_bsp = np.where(subset['Win']==1, 10*(subset['BSP']-1)*0.92, -10).sum()
        roi_bsp = ret_bsp / (count*10) * 100
        
        diff = roi_early - roi_bsp
        marker = "!!!" if diff > 0 else ""
        
        print(f"{label:<12} | {count:<6} | {steam_rate:>5.1f}% | {roi_early:>+8.1f}%   | {roi_bsp:>+7.1f}% {marker}")
        
    # Write to file for safety
    with open('value_steamers.txt', 'w') as f:
        f.write(f"{'Edge Bin':<12} | {'Bets':<6} | {'Steam%':<6} | {'Early ROI':<10} | {'BSP ROI':<8}\n")
        f.write("-" * 60 + "\n")
        for label in labels:
            subset = df_clean[df_clean['EdgeBin'] == label].copy()
            if len(subset) < 100: continue
            count = len(subset)
            steam_rate = subset['Steamed'].mean() * 100
            subset['Win'] = (subset['win']==1).astype(int)
            ret_early = np.where(subset['Win']==1, 10*(subset['Price5Min']-1)*0.92, -10).sum()
            roi_early = ret_early / (count*10) * 100
            ret_bsp = np.where(subset['Win']==1, 10*(subset['BSP']-1)*0.92, -10).sum()
            roi_bsp = ret_bsp / (count*10) * 100
            diff = roi_early - roi_bsp
            marker = "!!!" if diff > 0 else ""
            f.write(f"{label:<12} | {count:<6} | {steam_rate:>5.1f}% | {roi_early:>+8.1f}%   | {roi_bsp:>+7.1f}% {marker}\n")

    print("-" * 60)
    
    # Deep Dive into "Positive" or "High" Edge bins if any show promise
    # Look for secondary features: Box, Grade, Track
    
    target_group = df_clean[(df_clean['Edge'] > 0.10) & (df_clean['Edge'] < 0.30)].copy()
    if not target_group.empty:
        print("\nDEEP DIVE (Edge 0.10 - 0.30)")
        print("Checking Box Bias for Steamers...")
        print(target_group.groupby('Box')['Steamed'].mean())
        
        # Track Check
        print("\nTop 5 Tracks for Steaming (Min 50 bets):")
        track_grp = target_group.groupby('TrackName').agg({'Steamed': ['count', 'mean']})
        track_grp.columns = ['Count', 'SteamRate']
        track_grp = track_grp[track_grp['Count'] > 50].sort_values('SteamRate', ascending=False)
        print(track_grp.head(5))

if __name__ == "__main__":
    find_value_steamers()
