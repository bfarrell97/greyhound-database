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

def check_track_steamers():
    print("Loading Data (2025) for Track Steamer Check...")
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
    
    # Filter for Positive Edge Strategy (Option C/B/A pool)
    value_bets = df_clean[df_clean['Edge'] > 0.15].copy()
    
    print("\n" + "="*80)
    print("TRACK STEAMER CHECK (Positive Edge Bets Only)")
    print(f"Total Bets: {len(value_bets)}")
    print("="*80)
    
    # Group by Track
    # We want tracks where Early ROI > BSP ROI
    
    results = []
    
    for track, sub in value_bets.groupby('TrackName'):
        if len(sub) < 50: continue
            
        count = len(sub)
        sub['Win'] = (sub['win']==1).astype(int)
        
        ret_early = np.where(sub['Win']==1, 10*(sub['Price5Min']-1)*0.92, -10).sum()
        roi_early = ret_early / (count*10) * 100
        
        ret_bsp = np.where(sub['Win']==1, 10*(sub['BSP']-1)*0.92, -10).sum()
        roi_bsp = ret_bsp / (count*10) * 100
        
        diff = roi_early - roi_bsp
        
        results.append({
            'Track': track,
            'Bets': count,
            'EarlyROI': roi_early,
            'BSPROI': roi_bsp,
            'Diff': diff
        })
        
    results.sort(key=lambda x: x['Diff'], reverse=True)
    
    with open('track_steamers.txt', 'w') as f:
        f.write(f"{'Track':<15} | {'Bets':<6} | {'Early ROI':<10} | {'BSP ROI':<10} | {'Diff':<6}\n")
        f.write("-" * 65 + "\n")
        for r in results: 
            if r['Diff'] > 0 and r['EarlyROI'] > 0:
                marker = "!!!" 
                f.write(f"{r['Track']:<15} | {r['Bets']:<6} | {r['EarlyROI']:>+8.1f}% | {r['BSPROI']:>+8.1f}% | {r['Diff']:>+.1f}% {marker}\n")

    print("Written profitable steamer tracks to track_steamers.txt")

    # Check if ANY track has Diff > 0 and EarlyROI > 0
    winners = [r for r in results if r['Diff'] > 0 and r['EarlyROI'] > 0]
    if winners:
        print("\nFOUND TRACKS WHERE EARLY PRICE WINS AND IS PROFITABLE:")
        for w in winners:
             print(f"- {w['Track']} (Early ROI: {w['EarlyROI']:.1f}%)")
    else:
        print("\nNo tracks found where Early Price is better AND profitable.")

if __name__ == "__main__":
    check_track_steamers()
