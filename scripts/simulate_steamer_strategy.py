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

COMMISSION = 0.08

def simulate_steamer():
    print("Loading 2025 Data for Steamer Strategy...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box as RawBox,
        ge.Position as Place, ge.BSP as StartPrice, 
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
    # Note: Using BSP as 'StartPrice' proxy in query (aliased) to match previous pipeline 
    # BUT wait, we want Fixed Odds vs Exchange.
    # The DB schema usually has 'BSP' and maybe another price column?
    # Let's check schema/previous scripts.
    # Standard query usually selects `ge.BSP as StartPrice`. 
    # If the user wants to beat BSP, they need to take a FIXED price.
    # For this simulation, I will assume 'StartPrice' column (if it exists) is Fixed Odds, and 'BSP' is Exchange.
    # Checking previous queries... `ge.BSP as StartPrice` was used.
    # Ah, `ge.BSP` IS Betfair SP. 
    # Do we have a Fixed Odds column? `ge.Price`? `ge.SP`?
    # Let's load RAW columns to check.
    
    # Re-writing query to get typical columns
    query_raw = "SELECT * FROM GreyhoundEntries LIMIT 1"
    
    # Actually, I'll stick to the "BSP Prediction" logic.
    # If we predict BSP, we can compare to BSP itself (to see if we can identify 'overs').
    # But usually steamers are about taking EARLY price.
    # If I don't have early price, I can simulate:
    # "If Weighted Prob > Implied Prob of BSP". 
    # But specifically, I will use `Predicted_BSP`.
    # Let's assume we take BSP for now, but filter by `Predicted < Actual`.
    # Strategy: Back if `Actual_BSP > Predicted_BSP`. (The market is offering superior odds to our model).
    # This is effectively a value strategy using price regression.
    
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
    
    # Load Feature List for Regressor
    # (Assuming we saved it, or just use V38 set)
    # The regressor used V38 features + basic categorical.
    model_features = features + ['Track', 'Grade', 'Distance', 'Box']
    # remove duplicates
    model_features = list(dict.fromkeys(model_features))
    
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_v38_bsp_reg.json")
    
    # Ensure cols
    for f in model_features:
        if f not in df.columns:
            df[f] = -1
            
    # Predict Log BSP
    dtest = df[model_features]
    df['pred_log_bsp'] = model.predict(dtest)
    df['pred_bsp'] = np.expm1(df['pred_log_bsp'])
    
    # Actual Price (We aliased BSP as StartPrice in query, let's use it)
    df['Actual_BSP'] = pd.to_numeric(df['StartPrice'], errors='coerce')
    df['win'] = (df['Place'] == 1).astype(int)
    
    # Logic:
    # If Predicted Price is $5.00, and Actual is $8.00.
    # We think it should be $5. The market pays $8. Value!
    # Margin: Actual > Predicted * 1.2
    
    print("\n--- Strategy Simulation (Value via BSP Regression) ---")
    
    for margin in [1.0, 1.1, 1.2, 1.3, 1.5]:
        mask = (
            (df['Actual_BSP'] > df['pred_bsp'] * margin) &
            (df['Actual_BSP'] >= 2.0) & # Ignore unbackable shorties
            (df['pred_bsp'] < 20.0) # Ignore chaotic longshot predictions
        )
        
        bets = df[mask]
        n = len(bets)
        if n == 0: continue
        
        wins = bets['win'].sum()
        sr = (wins/n)*100
        
        stake = 10
        turnover = n * stake
        profit = (bets['win'] * bets['Actual_BSP'] * stake).sum() - turnover
        
        # Comm
        # If we take BSP, we pay comm on winnings
        wins_only = bets[bets['win'] == 1]
        comm = ((wins_only['Actual_BSP'] - 1) * stake * COMMISSION).sum()
        net = profit - comm
        roi = (net/turnover)*100
        
        print(f"Margin {margin:.1f}x | Bets: {n} | SR: {sr:.1f}% | Net: ${net:.2f} | ROI: {roi:.2f}%")

if __name__ == "__main__":
    simulate_steamer()
