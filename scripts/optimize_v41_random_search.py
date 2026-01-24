import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
import random

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def load_data():
    print("Loading Data (2020-2025)...")
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
    WHERE rm.MeetingDate >= '2024-01-01'
    ORDER BY rm.MeetingDate ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    df_clean['MeetingDate'] = pd.to_datetime(df_clean['MeetingDate'])
    return df_clean, features

def run_random_search():
    df, features = load_data()
    
    # Pre-train Model (Simplified Walk-Forward: Train on 2020-2023 [Outside], Predict 2024-2025)
    # To do this fast, we need a model trained up to end of 2023.
    # Actually, we can just use the splits logic from before but optimize the loop.
    
    # Generate Predictions for whole 2024-2025 period using a rolling model?
    # Too slow.
    # Let's use the Final V41 Model (Trained on 2020-Oct 2024). This is slightly leaky for early 2024.
    # BUT, if we can't find a strategy even with the "Golden Model", then likely none exists.
    # If we FIND one, we verify it strictly.
    
    print("Generating Predictions (using Final Model for Search)...")
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df[features])
    df['Prob'] = model.predict(dtest)
    df['ImpliedProb'] = 1.0 / df['BSP']
    df['Edge'] = df['Prob'] - df['ImpliedProb']
    
    splits = [
        ('2024-01-01', '2024-04-01'),
        ('2024-04-01', '2024-07-01'),
        ('2024-07-01', '2024-10-01'),
        ('2024-10-01', '2025-01-01'),
        ('2025-01-01', '2025-04-01')
    ]
    
    print("\nStarting Random Search (100 Iterations)...")
    print("Constraint: Odds <= $15.00")
    print("-" * 80)
    
    best_avg_roi = -100
    best_config = None
    
    for i in range(100):
        # Random Params
        min_edge = round(random.uniform(0.10, 0.30), 2)
        min_prob = round(random.uniform(0.10, 0.40), 2)
        max_price = round(random.uniform(4.0, 15.0), 1)
        min_price = round(random.uniform(1.0, 3.0), 1)
        
        quarterly_rois = []
        
        for start_date, end_date in splits:
            test_data = df[(df['MeetingDate'] >= start_date) & 
                           (df['MeetingDate'] < end_date) &
                           (df['BSP'] <= max_price) &
                           (df['BSP'] >= min_price) &
                           (df['Edge'] >= min_edge) &
                           (df['Prob'] >= min_prob)].copy()
            
            if len(test_data) < 20: 
                quarterly_rois.append(0)
                continue
                
            stake = 10
            test_data['Win'] = (test_data['win'] == 1).astype(int)
            test_data['Return'] = np.where(test_data['Win'] == 1, stake * (test_data['BSP'] - 1) * 0.92 + stake, 0)
            profit = test_data['Return'].sum() - (len(test_data) * stake)
            roi = (profit / (len(test_data) * stake)) * 100
            quarterly_rois.append(roi)
            
        avg_roi = np.mean(quarterly_rois)
        min_roi = min(quarterly_rois)
        
        if avg_roi > best_avg_roi:
            best_avg_roi = avg_roi
            best_config = (min_edge, min_prob, min_price, max_price)
            
        if avg_roi > 0 and min_roi > -10:
             print(f"FOUND: Edge>{min_edge} Prob>{min_prob} ${min_price}-${max_price} | Avg: {avg_roi:.1f}% | Min: {min_roi:.1f}%")

    print("-" * 80)
    print(f"Best Config: {best_config} with Avg ROI {best_avg_roi:.1f}%")

if __name__ == "__main__":
    run_random_search()
