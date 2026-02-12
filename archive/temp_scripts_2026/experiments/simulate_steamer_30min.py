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

def simulate_steamer_with_30min():
    print("Loading 2025 Data for Steamer Strategy (30min Prices)...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Check if Price30Min column exists first (to avoid query error)
    cursor = conn.execute("PRAGMA table_info(GreyhoundEntries)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'Price30Min' not in columns:
        print("Error: 'Price30Min' column not found in GreyhoundEntries table.")
        return
        
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box as RawBox,
        ge.Position as Place, ge.BSP, 
        ge.Price30Min, -- The Early Price
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
    AND ge.Price30Min > 0 
    """ 
    # Only select rows where we actually have a captured 30min price
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows with 30-min Price Data.")
    
    if df.empty:
        print("No data available for valid 30-min price simulation.")
        return
        
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
    
    # Features for Model
    model_features = features + ['Track', 'Grade', 'Distance', 'Box']
    model_features = list(dict.fromkeys(model_features))
    
    # Load BSP Model
    # Try optimized first, else baseline
    model = xgb.XGBRegressor()
    try:
        model.load_model("models/xgb_v38_bsp_reg_optimized.json")
    except:
        try:
            model.load_model("models/xgb_v38_bsp_reg.json")
        except:
             print("Model not found.")
             return
             
    # Ensure cols
    for f in model_features:
        if f not in df.columns:
            df[f] = -1
            
    # Predict Log BSP
    dtest = df[model_features]
    df['pred_log_bsp'] = model.predict(dtest)
    df['Predicted_BSP'] = np.expm1(df['pred_log_bsp'])
    
    df['win'] = (df['Place'] == 1).astype(int)
    
    print("\n--- Model Bias Check ---")
    print(f"Mean Price30Min:  ${df['Price30Min'].mean():.2f}")
    print(f"Mean BSP:         ${df['BSP'].mean():.2f}")
    print(f"Mean Predicted:   ${df['Predicted_BSP'].mean():.2f}")
    print(f"Model Bias:       {df['Predicted_BSP'].mean() - df['BSP'].mean():.2f} (Pos = Overpredicting, Neg = Underpredicting)")
    
    # Correction Factor
    # We want Mean(Pred) ~= Mean(BSP)
    bias_factor = df['BSP'].mean() / df['Predicted_BSP'].mean()
    print(f"Bias Correction Factor: {bias_factor:.3f}")
    df['Predicted_Corrected'] = df['Predicted_BSP'] * bias_factor
    
    print("\n--- Strategy Simulation: Backing Price30Min if Predicted To Steam (Bias Corrected) ---")
    print("Logic: If Predicted_Corrected < Price30Min * 0.X")
    
    # Load Classifier for Ranking
    print("Loading Classifier for Rank...")
    classifier = xgb.Booster()
    try:
        classifier.load_model("models/xgb_v38_beyer_2024.json")
    except:
        print("Classifier model not found. Cannot rank.")
        return

    # Ensure classifier features exist
    with open('models/v38_features.json', 'r') as f:
        clf_features = json.load(f)
        
    for f in clf_features:
        if f not in df.columns:
            df[f] = -1
            
    dtest_clf = xgb.DMatrix(df[clf_features])
    df['prob_win'] = classifier.predict(dtest_clf)
    
    df['rank'] = df.groupby('RaceID')['prob_win'].rank(ascending=False, method='first')

    print("\n--- Detailed Analysis by Price Band (< $15) + Rank 1 Filter ---")
    
    price_bands = [(2.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
    
    for min_p, max_p in price_bands:
        print(f"\nBand ${min_p}-{max_p}:")
        for thresh in [0.8, 0.9, 0.95]:
            mask = (
                (df['Predicted_Corrected'] < df['Price30Min'] * thresh) &
                (df['Price30Min'] >= min_p) &
                (df['Price30Min'] < max_p) &
                (df['rank'] == 1)
            )
            
            bets = df[mask]
            n = len(bets)
            if n < 5: continue
            
            wins = bets['win'].sum()
            sr = (wins/n)*100
            
            stake = 10
            turnover = n * stake
            profit = (bets['win'] * bets['Price30Min'] * stake).sum() - turnover
            
            roi = (profit/turnover)*100
            
            print(f"  Rank 1 & Pred < {thresh}x | Bets: {n} | SR: {sr:.1f}% | ROI: {roi:.1f}%")

if __name__ == "__main__":
    simulate_steamer_with_30min()
