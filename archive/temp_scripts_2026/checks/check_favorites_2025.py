import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
import json
import os
import sys

# Add project root
sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v38 import FeatureEngineerV38
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from features.feature_engineering_v38 import FeatureEngineerV38

COMMISSION = 0.05 # Use 5% for standard comparison

def check_favorites():
    print("Checking V38 Rank 1 Performance on 2025 Favorites...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # We use BSP for PnL
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box as RawBox,
        ge.Position as Place, ge.BSP, 
        ge.Split, ge.FinishTime, ge.Weight, ge.BeyerSpeedFigure, ge.InRun,
        r.Distance, r.Grade, r.RaceTime,
        t.TrackName as RawTrack, rm.MeetingDate as date_dt,
        ge.TrainerID
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2025-01-01' 
    AND ge.FinishTime > 0
    AND ge.BSP > 0
    """ 
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Feature Engineering
    fe = FeatureEngineerV38()
    df, features = fe.engineer_features(df)
    
    # Mappings
    with open('models/v33_mappings.json', 'r') as f:
        mappings = json.load(f)
    for col in ['Track', 'Grade', 'Distance']:
        target_col = 'RawTrack' if col == 'Track' else col
        col_map = mappings.get(col, {})
        df[col] = df[target_col].astype(str).map(col_map).fillna(-1).astype(int)
    df['Box'] = pd.to_numeric(df['RawBox'], errors='coerce').fillna(0).astype(int)
    
    # Load Classifier
    classifier = xgb.Booster()
    classifier.load_model("models/xgb_v38_beyer_2024.json")
    
    with open('models/v38_features.json', 'r') as f:
        clf_features = json.load(f)
    
    for f in clf_features:
        if f not in df.columns:
            df[f] = -1
            
    dtest = xgb.DMatrix(df[clf_features])
    df['prob_win'] = classifier.predict(dtest)
    
    # Rank
    df['rank'] = df.groupby('RaceID')['prob_win'].rank(ascending=False, method='first')
    df['win'] = (df['Place'] == 1).astype(int)
    
    # Analyze Favorites in Bands
    print("\n--- PnL Analysis (Commission 5%) ---")
    
    bands = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 10.0)]
    
    for min_p, max_p in bands:
        # Bets: Rank 1 AND Price in Band
        mask = (df['rank'] == 1) & (df['BSP'] >= min_p) & (df['BSP'] < max_p)
        bets = df[mask]
        n = len(bets)
        if n == 0: continue
        
        wins = bets['win'].sum()
        sr = (wins/n)*100
        
        stake = 10
        turnover = n * stake
        # Profit
        gross_profit = (bets['win'] * bets['BSP'] * stake).sum()
        comm = (gross_profit - stake * wins) * COMMISSION # Comm on winnings
        net = gross_profit - turnover - comm
        
        roi = (net/turnover)*100
        
        print(f"Rank 1 @ ${min_p}-{max_p} | Bets: {n} | SR: {sr:.1f}% | Net: ${net:.2f} | ROI: {roi:.2f}%")

if __name__ == "__main__":
    check_favorites()
