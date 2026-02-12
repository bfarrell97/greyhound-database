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

def analyze_open_prices():
    print("Loading Data (2024-2025) for Marginal Volume Analysis...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.PriceOpen, ge.Price5Min, 
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
    
    # Predict
    model = joblib.load('models/xgb_v41_final.pkl')
    df_pred = df.dropna(subset=features).copy()
    dtest = xgb.DMatrix(df_pred[features])
    df_pred['Prob'] = model.predict(dtest)
    df_pred['ImpliedProb'] = 1.0 / df_pred['BSP']
    df_pred['Edge'] = df_pred['Prob'] - df_pred['ImpliedProb']
    
    # Filter for Valid Prices
    df_pred = df_pred[
        (df_pred['BSP'] > 1.0) & 
        (df_pred['PriceOpen'] > 1.0)
    ].copy()
    
    print("\n" + "="*80)
    print("OPEN PRICE vs BSP ANALYSIS (Searching for 'Steamer' Strategy)")
    print("Constraint: High Volume, Marginal BSP ROI")
    print("="*80)
    
    # Grid Search for "Marginal" Strategy
    configs = [
        {'edge': 0.10, 'prob': 0.10, 'price': 15.0},
        {'edge': 0.15, 'prob': 0.10, 'price': 15.0},
        {'edge': 0.10, 'prob': 0.10, 'price': 10.0},
        {'edge': 0.05, 'prob': 0.10, 'price': 15.0}, # Very loose
        {'edge': 0.10, 'prob': 0.15, 'price': 15.0},
    ]
    
    for c in configs:
        bets = df_pred[
            (df_pred['BSP'] <= c['price']) &
            (df_pred['Edge'] >= c['edge']) &
            (df_pred['Prob'] >= c['prob'])
        ].copy()
        
        if len(bets) < 100: continue
            
        bets['Win'] = (bets['win'] == 1).astype(int)
        stake = 10
        
        # BSP Return
        bets['Ret_BSP'] = np.where(bets['Win']==1, stake*(bets['BSP']-1)*0.92+stake, 0)
        prof_bsp = bets['Ret_BSP'].sum() - (len(bets)*stake)
        roi_bsp = (prof_bsp / (len(bets)*stake)) * 100
        
        # Open Return
        bets['Ret_Open'] = np.where(bets['Win']==1, stake*(bets['PriceOpen']-1)*0.92+stake, 0)
        prof_open = bets['Ret_Open'].sum() - (len(bets)*stake)
        roi_open = (prof_open / (len(bets)*stake)) * 100
        
        # Steamer Stats
        steamers = bets[bets['PriceOpen'] > bets['BSP']] # Opened higher, steamed in
        steamer_pct = len(steamers) / len(bets) * 100
        
        # Vol Calculation
        months = 15 # Approx
        vol_mo = len(bets) / months
        
        print(f"Cfg: Edge>{c['edge']} Prob>{c['prob']} <${c['price']}")
        print(f"  Vol: {len(bets)} ({int(vol_mo)}/mo) | Steamer%: {steamer_pct:.1f}%")
        print(f"  BSP ROI:  {roi_bsp:>6.1f}% (${prof_bsp:.0f})")
        print(f"  Open ROI: {roi_open:>6.1f}% (${prof_open:.0f})")
        print(f"  Diff:     {roi_open - roi_bsp:+.1f}%")
        print("-" * 40)

if __name__ == "__main__":
    analyze_open_prices()
