
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "greyhound_racing.db"
MODEL_PATH = "models/xgb_v33_test_2020_2024.json"
MAPPING_PATH = "models/v33_mappings.json"
INITIAL_BANK = 1000.0
COMMISSION = 0.08
LAY_TARGET_PCT = 0.06
BACK_STAKE = 10.0 # Flat stake for Backing

# TRACKS
SA_TRACKS = ['Angle Park', 'Gawler', 'Mount Gambier', 'Murray Bridge', 'Murray Bridge (MBR)']
VIC_TRACKS = [
    'Ballarat', 'Bendigo', 'Sale', 'Sandown Park', 'Warrnambool', 'Warragul', 
    'Geelong', 'Horsham', 'The Meadows', 'Shepparton', 'Traralgon', 'Cranbourne'
]
STRAIGHT_TRACKS = ['Healesville', 'Murray Bridge (MBS)', 'Richmond (RIS)', 'Richmond Straight', 'Capalaba', 'Q Straight']

def get_data_2025():
    print("[1/4] Loading 2025 Data...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box as RawBox,
        ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.Margin as Margin1,
        ge.BSP as StartPrice, ge.Weight,
        r.Distance, r.Grade, t.TrackName as RawTrack, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-10-01'
    AND ge.FinishTime > 0
    AND ge.BSP > 1.0
    ORDER BY rm.MeetingDate, r.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df = df.replace([np.inf, -np.inf], 0)
    
    # Feature Eng (Exact Match)
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
        
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            lag_name = f'{col}_Lag{i}'
            if col == 'StartPrice':
                 df[lag_name] = df.groupby('GreyhoundID')['StartPrice'].shift(i)
            else:
                 df[lag_name] = 0.0

    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed_avg'] = 0.0
    
    import json
    with open(MAPPING_PATH, 'r') as f:
        mappings = json.load(f)
        
    df['Track'] = df['RawTrack']
    df['Box'] = df['RawBox']
    for col in ['Track', 'Grade', 'Box', 'Distance']:
        col_map = mappings.get(col, {})
        df[col] = df[col].astype(str).map(col_map).fillna(-1)

    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    df['Weight'] = 30.0
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df[df['date_dt'] >= '2025-01-01'].copy()
    return df, feature_cols

def generate_bets(df):
    print("[3/4] Applying SOUTHERN SKEPTIC Strategy Filters...")
    bets = []
    
    df = df.sort_values(['date_dt', 'RaceID'])
    
    for _, row in df.iterrows():
        track = row['RawTrack']
        price = row['StartPrice']
        if price <= 1.0: continue
        
        prob = row['Prob']
        implied = 1/price
        edge = prob - implied
        box = row['RawBox']
        
        if row['Place_Lag2'] == -1: continue 
        
        bet_type = None
        
        # 1. LAY STRATEGIES (Survivors)
        if track in SA_TRACKS:
            if box in [1, 2, 3, 8] and price < 3.5 and prob < 0.15 and edge < -0.4:
                bet_type = 'LAY'
        elif track in VIC_TRACKS:
            if box in [1, 2, 3, 8] and price < 3.5 and prob < 0.25 and edge < -0.4:
                bet_type = 'LAY'
        elif track in STRAIGHT_TRACKS:
            if box in [1, 2, 3, 8] and price < 3.5 and prob < 0.15 and edge < -0.55:
                bet_type = 'LAY'
                
        # 2. BACK STRATEGY (SA ONLY - Value)
        # Prob > 20%, Price > 4.0, Edge > 5%
        # Note: Can't back and lay same dog. Prioritize LAY?
        # Actually logic ensures they are mutually exclusive.
        # Lay requires Price < 3.5. Back requires Price > 4.0.
        
        if not bet_type:
            if track in SA_TRACKS:
                if prob > 0.20 and price >= 4.0 and edge > 0.05:
                    bet_type = 'BACK'
                
        if bet_type:
            bets.append({
                'date': row['date_dt'],
                'type': bet_type,
                'price': price,
                'win': row['win']
            })
            
    return pd.DataFrame(bets)

def analyze_strategy():
    df, feature_cols = get_data_2025()
    
    print("[2/4] Predicting...")
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    dmat = xgb.DMatrix(df[feature_cols])
    df['Prob'] = booster.predict(dmat)
    
    bets_df = generate_bets(df)
    print(f"\nTotal Confirmed Bets: {len(bets_df)}")
    
    # Analyze by Type
    print("\n[Breakdown by Type]")
    print(bets_df.groupby('type')['price'].count())
    
    # Run continuous simulation
    print("\n[Running Simulation...]")
    bets_df = bets_df.sort_values('date')
    
    bank = INITIAL_BANK
    history = []
    
    for _, row in bets_df.iterrows():
        price = row['price']
        
        if row['type'] == 'LAY':
            target = bank * LAY_TARGET_PCT
            stake = target / (1 - COMMISSION)
            liability = stake * (price - 1)
            
            if liability > bank: # Cap liability
                liability = bank
                # stake = liability / (price - 1) # Implied stake reduced
            
            if row['win'] == 0: # Lay Win
                profit = stake * (1 - COMMISSION)
            else: # Lay Loss
                profit = -liability
                
        else: # BACK
            stake = BACK_STAKE # Flat $10
            if row['win'] == 1:
                profit = stake * (price - 1) * (1 - COMMISSION)
            else:
                profit = -stake
                
        bank += profit
        if bank < 0: bank = 0
        history.append(bank)
        
    print(f"Final Bank: ${bank:.2f}")
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(history)
    plt.title("Southern Skeptic Strategy - 2025 Growth")
    plt.ylabel("Bank ($)")
    plt.xlabel("Bets")
    plt.grid(True)
    plt.savefig("southern_skeptic_chart.png")

if __name__ == "__main__":
    analyze_strategy()
