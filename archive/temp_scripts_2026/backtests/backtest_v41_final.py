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

def backtest_v41():
    print("Loading Jan 2025 Data for V41 Backtest...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Same query as V41 training
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, t.TrackName, rm.MeetingDate,
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
    
    # Edge
    df['ImpliedProb'] = 1.0 / df['BSP']
    df['Edge'] = df['Prob'] - df['ImpliedProb']
    
    # Thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print("\n-----------------------------------------------------------")
    print("BACKTEST: V41 SUPER MODEL (Jan 2025)")
    print("-----------------------------------------------------------")
    print(f"{'Min Edge':<10} | {'Bets':<6} | {'Strike':<6} | {'Turnover':<10} | {'Profit':<10} | {'ROI':<6}")
    print("-" * 80)
    
    best_roi = -100
    
    for t in thresholds:
        bets = df[df['Edge'] > t].copy()
        
        if len(bets) == 0:
            continue
            
        bets['Win'] = (bets['win'] == 1).astype(int)
        stake = 10
        bets['Return'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
        
        turnover = len(bets) * stake
        profit = bets['Return'].sum() - turnover
        roi = (profit / turnover) * 100
        strike_rate = (bets['Win'].sum() / len(bets)) * 100
        
        print(f"{t:>8.2f}   | {len(bets):>6} | {strike_rate:>5.1f}% | ${turnover:>9,.0f} | ${profit:>9,.2f} | {roi:>5.1f}%")
        
    print("-" * 80)
    
    # Interaction Check: High Edge + Vacant Box + High Grade?
    # Actually, model should handle it. Let's check a simple 'Smart Money' filter
    # Filter: Edge > 0.10 AND (Trainer_Track_Rate > 0.15)
    
    smart = df[(df['Edge'] > 0.10) & (df['Trainer_Track_Rate'] > 0.15)].copy()
    if len(smart) > 0:
        smart['Win'] = (smart['win'] == 1).astype(int)
        stake = 10
        smart['Return'] = np.where(smart['Win'] == 1, stake * (smart['BSP'] - 1) * 0.92 + stake, 0)
        profit = smart['Return'].sum() - (len(smart)*stake)
        roi = (profit / (len(smart)*stake)) * 100
        print(f"FILTER (Edge>0.10 & HighTrainer): Bets {len(smart)}, ROI {roi:.2f}%")

if __name__ == "__main__":
    backtest_v41()
