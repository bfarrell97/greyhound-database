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

def profile_steamers():
    print("Loading Data (2025) for Steamer Profiling...")
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
    
    print(f"Loaded {len(df)} rows with Price5Min data.")
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    
    # Predict V41
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df_clean[features])
    df_clean['Prob'] = model.predict(dtest)
    df_clean['ImpliedProb'] = 1.0 / df_clean['BSP']
    df_clean['Edge'] = df_clean['Prob'] - df_clean['ImpliedProb']
    
    # Define Steamer: BSP < Price5Min * 0.90 (Dropped 10%)
    df_clean['IsSteamer'] = (df_clean['BSP'] < (df_clean['Price5Min'] * 0.90)).astype(int)
    # Define Drifter: BSP > Price5Min * 1.10 (Rose 10%)
    df_clean['IsDrifter'] = (df_clean['BSP'] > (df_clean['Price5Min'] * 1.10)).astype(int)
    
    steamers = df_clean[df_clean['IsSteamer']==1]
    drifters = df_clean[df_clean['IsDrifter']==1]
    stable = df_clean[(df_clean['IsSteamer']==0) & (df_clean['IsDrifter']==0)]
    
    print("\n" + "="*80)
    print("STEAMER PROFILE ANALYSIS (2025 Data)")
    print(f"Total: {len(df_clean)} | Steamers (10%+ Drop): {len(steamers)}")
    print("="*80)
    
    # Correlations
    cols_to_check = ['Prob', 'Edge', 'BSP', 'Price5Min', 'Box', 'Weight', 'DogAgeDays', 'RunTimeNorm_Lag1']
    
    print(f"{'Metric':<20} | {'Steamers':<10} | {'Drifters':<10} | {'Stable':<10}")
    print("-" * 60)
    
    for c in cols_to_check:
        if c in df_clean.columns:
            s_val = steamers[c].mean()
            d_val = drifters[c].mean()
            n_val = stable[c].mean()
            print(f"{c:<20} | {s_val:>10.4f} | {d_val:>10.4f} | {n_val:>10.4f}")
            
    print("-" * 60)
    print("Win Rate Comparison:")
    print(f"{'Win Rate':<20} | {steamers['win'].mean()*100:>9.1f}% | {drifters['win'].mean()*100:>9.1f}% | {stable['win'].mean()*100:>9.1f}%")
    
    # Edge Bin Analysis
    print("\n--- Edge vs Steam Probability ---")
    df_clean['EdgeBin'] = pd.cut(df_clean['Edge'], bins=[-1, -0.1, 0, 0.1, 0.2, 0.3, 1.0])
    grouped = df_clean.groupby('EdgeBin')['IsSteamer'].mean()
    print(grouped)

if __name__ == "__main__":
    profile_steamers()
