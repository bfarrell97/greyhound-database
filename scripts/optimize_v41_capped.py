import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
from itertools import product

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def optimize_capped():
    print("Loading Jan 2025 Data for Optimization...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.BSP > 1.0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    # Predict
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df[features])
    df['Prob'] = model.predict(dtest)
    df['ImpliedProb'] = 1.0 / df['BSP']
    df['Edge'] = df['Prob'] - df['ImpliedProb']
    df['Win'] = (df['win'] == 1).astype(int)
    
    # Grid Search Space
    # Constraint: BSP <= 15.0
    df = df[df['BSP'] <= 15.0].copy()
    
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    probs = [0.0, 0.10, 0.20, 0.30, 0.40]
    max_price_caps = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0] # Test tighter caps
    
    results = []
    
    print("\n" + "="*80)
    print("GRID SEARCH OPTIMIZATION (V41, BSP <= $15.00)")
    print("Objective: Find profitable sub-segments.")
    print("Comm: 8%")
    print("="*80)
    
    for edge_min, prob_min, price_cap in product(edges, probs, max_price_caps):
        # Filter
        bets = df[
            (df['Edge'] >= edge_min) & 
            (df['Prob'] >= prob_min) & 
            (df['BSP'] <= price_cap) & 
            (df['BSP'] >= 1.50) # Exclude unbackable shorts
        ].copy()
        
        if len(bets) < 100: # Ignore tiny samples
            continue
            
        stake = 10
        bets['Return'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
        profit = bets['Return'].sum() - (len(bets) * stake)
        roi = (profit / (len(bets) * stake)) * 100
        strike = bets['Win'].mean() * 100
        
        results.append({
            'MinEdge': edge_min,
            'MinProb': prob_min,
            'MaxPrice': price_cap,
            'Bets': len(bets),
            'Strike': strike,
            'Profit': profit,
            'ROI': roi
        })
        
    # Sort by Profit
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No profitable configurations found.")
        return

    top_configs = results_df.sort_values('Profit', ascending=False).head(20)
    
    print(f"{'Edge >':<6} | {'Prob >':<6} | {'Price <':<7} | {'Bets':<6} | {'Strike':<6} | {'Profit':<10} | {'ROI':<6}")
    print("-" * 80)
    
    for _, r in top_configs.iterrows():
        print(f"{r['MinEdge']:>6.2f} | {r['MinProb']:>6.2f} | ${r['MaxPrice']:>6.2f} | {int(r['Bets']):>6} | {r['Strike']:>5.1f}% | ${r['Profit']:>9.2f} | {r['ROI']:>5.1f}%")

if __name__ == "__main__":
    optimize_capped()
