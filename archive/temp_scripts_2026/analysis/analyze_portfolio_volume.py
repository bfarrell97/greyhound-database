
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
DB_PATH = "greyhound_racing.db"
MAPPING_PATH = "models/v33_mappings.json"

MODELS = {
    'GENERAL': 'models/xgb_v33_test_2020_2024.json',
    'NSW': 'models/xgb_v33_nsw_2020_2024.json',
    'VIC': 'models/xgb_v33_vic_2020_2024.json',
    'SA': 'models/xgb_v33_sa_2020_2024.json',
    'WA': 'models/xgb_v33_wa_2020_2024.json',
    'STRAIGHT': 'models/xgb_v33_straight_2020_2024.json'
}

NSW_TRACKS = [
    'Gunnedah', 'Richmond', 'Taree', 'Temora', 'Casino', 'Nowra', 
    'Wentworth Park', 'The Gardens', 'Goulburn', 'Wagga', 'Grafton', 
    'Broken Hill', 'Maitland', 'Bulli', 'Gosford', 'Muswellbrook', 
    'Dubbo', 'Dapto', 'Bathurst', 'Lismore'
]
VIC_TRACKS = [
    'Ballarat', 'Bendigo', 'Sale', 'Sandown Park', 'Warrnambool', 'Warragul', 
    'Geelong', 'Horsham', 'The Meadows', 'Shepparton', 'Traralgon', 'Cranbourne'
]
SA_TRACKS = ['Angle Park', 'Gawler', 'Mount Gambier', 'Murray Bridge', 'Murray Bridge (MBR)']
WA_TRACKS = ['Cannington', 'Mandurah', 'Northam']
STRAIGHT_TRACKS = ['Healesville', 'Murray Bridge (MBS)', 'Richmond (RIS)', 'Richmond Straight', 'Capalaba', 'Q Straight']
QLD_CIRCULAR = ['Albion Park', 'Ipswich', 'Rockhampton', 'Townsville', 'Q1 Lakeside', 'Q2 Parklands', 'Bundaberg']

def get_data_2025():
    print("[1/2] Loading 2025 Data...")
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
    df['Margin1'] = df['Margin1'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Margin1'] = pd.to_numeric(df['Margin1'], errors='coerce').fillna(99.0)
    
    df = df.sort_values(['GreyhoundID', 'date_dt'])
    for i in range(1, 11):
        df[f'Place_Lag{i}'] = df.groupby('GreyhoundID')['Place'].shift(i)
    for col in ['StartPrice', 'RunTime', 'SplitMargin', 'Margin1']:
        for i in range(1, 4):
            df[f'{col}_Lag{i}'] = df.groupby('GreyhoundID')[col].shift(i)
    df['SR_avg'] = df.groupby('GreyhoundID')['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    df['RunSpeed'] = df['Distance'] / df['RunTime']
    df['RunSpeed_avg'] = df.groupby('GreyhoundID')['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
    
    with open(MAPPING_PATH, 'r') as f:
        mappings = json.load(f)
    categorical_cols = ['Track', 'Grade', 'Box', 'Distance']
    df['Track'] = df['RawTrack']
    df['Box'] = df['RawBox']
    for col in categorical_cols:
        col_map = mappings.get(col, {})
        df[col] = df[col].astype(str).map(col_map).fillna(-1)
        
    lag_cols = [c for c in df.columns if 'Lag' in c]
    df[lag_cols] = df[lag_cols].fillna(-1)
    feature_cols = lag_cols + ['SR_avg', 'RunSpeed_avg', 'Track', 'Grade', 'Box', 'Distance', 'Weight']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.replace([np.inf, -np.inf], 0)
    return df[df['date_dt'] >= '2025-01-01'].copy(), feature_cols

def analyze_volume():
    df, feature_cols = get_data_2025()
    
    # Load Models
    boosters = {}
    for name, path in MODELS.items():
        b = xgb.Booster()
        b.load_model(path)
        boosters[name] = b
        
    print("[2/2] Running Strategy Simulation...")
    df['Bet'] = False
    
    def apply_strategy(row):
        track = row['RawTrack']
        price = row['StartPrice']
        box = row['RawBox']
        has_hist = row['Place_Lag2'] != -1
        
        # 1. Determine Model
        if track in NSW_TRACKS: model_name = 'NSW'
        elif track in VIC_TRACKS: model_name = 'VIC'
        elif track in SA_TRACKS: model_name = 'SA'
        elif track in WA_TRACKS: model_name = 'WA'
        elif track in STRAIGHT_TRACKS: model_name = 'STRAIGHT'
        else: model_name = 'GENERAL'
        
        # 2. Predict
        # We do this race-by-race for simplicity in volume check
        # (Though vectorizing would be faster)
        return model_name
    
    # Vectorize predictions for efficiency
    for name, b in boosters.items():
        dmat = xgb.DMatrix(df[feature_cols])
        df[f'Prob_{name}'] = b.predict(dmat)
        
    def check_bet(row):
        track = row['RawTrack']
        price = row['StartPrice']
        if price <= 0: return False
        box = row['RawBox']
        has_hist = row['Place_Lag2'] != -1
        if not has_hist: return False
        
        # NSW
        if track in NSW_TRACKS:
            prob = row['Prob_NSW']
            edge = prob - (1/price)
            # LAY
            if box in [1,2,3,8] and price < 4.0 and prob < 0.20 and edge < -0.48: return True
            # BACK
            if prob > 0.55 and edge > 0.25: return True
            
        # VIC
        elif track in VIC_TRACKS:
            prob = row['Prob_VIC']
            edge = prob - (1/price)
            if box in [1,2,3,8] and price < 4.0 and prob < 0.20 and edge < -0.48: return True
            
        # SA
        elif track in SA_TRACKS:
            prob = row['Prob_SA']
            edge = prob - (1/price)
            if price < 4.0 and prob < 0.15 and edge < -0.55: return True
            
        # WA
        elif track in WA_TRACKS:
            prob = row['Prob_WA']
            edge = prob - (1/price)
            if prob > 0.55 and edge > 0.25: return True
            
        # STRAIGHT
        elif track in STRAIGHT_TRACKS:
            prob = row['Prob_STRAIGHT']
            edge = prob - (1/price)
            if prob > 0.55 and edge > 0.15: return True
            
        # OTHERS (Standard LAY + QLD BACK/LAY)
        else:
            prob = row['Prob_GENERAL']
            edge = prob - (1/price)
            # General LAY (QLD/TAS/NZ etc)
            if box in [1,2,3,8] and price < 4.0 and prob < 0.20 and edge < -0.48: return True
            # QLD Special BACK (Generic model excels here)
            if track in QLD_CIRCULAR and prob > 0.55 and edge > 0.15: return True
            
        return False

    df['IsBet'] = df.apply(check_bet, axis=1)
    bets = df[df['IsBet']]
    
    total_bets = len(bets)
    days = (df['date_dt'].max() - df['date_dt'].min()).days + 1
    avg_daily = total_bets / days
    
    daily_counts = bets.groupby('date_dt').size()
    
    print("\n" + "="*60)
    print("PORTFOLIO VOLUME ANALYSIS (2025)")
    print("="*60)
    print(f"Total Bets:       {total_bets}")
    print(f"Total Days:       {days}")
    print(f"Average Bets/Day: {avg_daily:.2f}")
    print(f"Max Bets/Day:     {daily_counts.max()}")
    print(f"Quiet Days (0):   {int(days - len(daily_counts))}")
    print("-" * 60)
    print("Weekly Distribution (Total / 7):")
    print(f"Avg Bets/Week:    {avg_daily * 7:.1f}")
    print("="*60)

if __name__ == "__main__":
    analyze_volume()
