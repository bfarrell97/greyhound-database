"""
AutoGluon V20 - Combined Historical + Live Price Model
Uses Top 30 Features from Factory + Price5Min (Live Market Data)
Target: LogBSP
"""
import sqlite3
import pandas as pd
import numpy as np
import time
import random
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("AUTOGLUON V20 - HYBRID PRICE MODEL")
print("Features: Top 30 History + Price5Min")
print("Target: LogBSP")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL 
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Year'] = df['MeetingDate'].dt.year
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))

# Clean Price5Min - Key Predictor
df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')
df['HasPrice5Min'] = df['Price5Min'].notna().astype(int)
df['LogPrice5Min'] = np.log(df['Price5Min'].fillna(df['BSP']).clip(1.01, 500)) # Fill missing with BSP for training? CAREFUL. 
# Better to fill missing P5 with BSP_Mean_10 or similar? 
# For now, we focus on rows THAT HAVE P5 for the 'Live Strategy' test.

print(f"[1/5] Loaded {len(df):,} entries")

print("[2/5] Building Top 30 Features...")

from collections import defaultdict
dog_hist = defaultdict(list)
trainer_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
sire_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
box_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})

rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    race_date = race_df['MeetingDate'].iloc[0]
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_hist.get(dog_id, [])
        
        if len(hist) >= 3:
            feat = {
                'RaceID': race_id,
                'LogBSP': r['LogBSP'], 
                'BSP': r['BSP'],
                'Year': r['Year'],
                'HasPrice5Min': r['HasPrice5Min'],
                'Price5Min': r['Price5Min'] if pd.notna(r['Price5Min']) else -1
            }
            
            # --- LIVE DATA FEATURE ---
            if feat['HasPrice5Min']:
                feat['LogPrice5Min'] = np.log(r['Price5Min'])
            else:
                feat['LogPrice5Min'] = 0 # Placeholder, model handles 0 if split
                
            # --- TOP 30 FEATURES FROM FACTORY ---
            
            # 1. BSP Rolling Stats (Mean, Max, Std, Min) for 3, 5, 10
            # We compute efficiently
            bsps = [h['bsp'] for h in hist[-10:]] # Max 10 needed
            pos = [h['pos'] for h in hist[-10:] if h['pos']]
            
            if len(bsps) >= 3:
                # 10 Race Stats
                feat['BSP_Mean_10'] = np.mean(bsps)
                feat['BSP_Max_10'] = max(bsps)
                feat['BSP_Min_10'] = min(bsps)
                feat['BSP_Std_10'] = np.std(bsps)
                
                # 5 Race Stats
                bsps5 = bsps[-5:]
                feat['BSP_Mean_5'] = np.mean(bsps5)
                feat['BSP_Max_5'] = max(bsps5)
                feat['BSP_Min_5'] = min(bsps5)
                feat['BSP_Std_5'] = np.std(bsps5)
                
                # 3 Race Stats
                bsps3 = bsps[-3:]
                feat['BSP_Mean_3'] = np.mean(bsps3)
                feat['BSP_Max_3'] = max(bsps3)
                feat['BSP_Min_3'] = min(bsps3)
                feat['BSP_Std_3'] = np.std(bsps3)
                
                # Position Stats
                if len(pos) >= 3:
                    feat['Pos_Mean_10'] = np.mean(pos)
                    feat['Pos_Mean_5'] = np.mean(pos[-5:])
                    feat['Pos_Mean_3'] = np.mean(pos[-3:])
                    feat['WinRate_10'] = sum(1 for p in pos if p==1)/len(pos)
                else:
                    feat['Pos_Mean_10'] = 4.5
                    feat['Pos_Mean_5'] = 4.5
                    feat['Pos_Mean_3'] = 4.5
                    feat['WinRate_10'] = 0
            
            # Context
            tid = r['TrainerID']
            if trainer_stat[tid]['runs'] > 5:
                feat['Trainer_AvgBSP'] = trainer_stat[tid]['bsp_sum'] / trainer_stat[tid]['runs']
            else:
                feat['Trainer_AvgBSP'] = 10
                
            sid = r['SireID']
            if sid and sire_stat[sid]['runs'] > 10:
                feat['Sire_AvgBSP'] = sire_stat[sid]['bsp_sum'] / sire_stat[sid]['runs']
            else:
                feat['Sire_AvgBSP'] = 10
                
            bid = (r['TrackID'], r['Box'])
            if box_stat[bid]['runs'] > 5:
                feat['Box_Track_AvgBSP'] = box_stat[bid]['bsp_sum'] / box_stat[bid]['runs']
            else:
                feat['Box_Track_AvgBSP'] = 10
            
            # Trend
            if len(hist) >= 5:
                feat['BSP_Trend_5'] = hist[-1]['bsp'] - hist[-5]['bsp']
            else:
                feat['BSP_Trend_5'] = 0
                
            rows.append(feat)
            
    # Update History
    for _, r in race_df.iterrows():
        dog_hist[r['GreyhoundID']].append({
            'bsp': r['BSP'] if r['BSP'] else 10,
            'pos': pd.to_numeric(r['Position'], errors='coerce')
        })
        
        # Stats
        bsp_val = r['BSP'] if r['BSP'] else 10
        trainer_stat[r['TrainerID']]['runs'] += 1
        trainer_stat[r['TrainerID']]['bsp_sum'] += bsp_val
        
        if r['SireID']:
            sire_stat[r['SireID']]['runs'] += 1
            sire_stat[r['SireID']]['bsp_sum'] += bsp_val
            
        box_stat[(r['TrackID'], r['Box'])]['runs'] += 1
        box_stat[(r['TrackID'], r['Box'])]['bsp_sum'] += bsp_val

    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

full_df = pd.DataFrame(rows)
# Split
train_df = full_df[full_df['Year'] <= 2023].copy()
test_df = full_df[full_df['Year'] >= 2024].copy()

# IMPORTANT: For training/testing the LIVE model, we must have Price5Min
# We filter datasets to only include rows with Price5Min to evaluate "Using Price5Min" performance
train_live = train_df[train_df['HasPrice5Min'] == 1].copy()
test_live = test_df[test_df['HasPrice5Min'] == 1].copy()

exclude = ['RaceID', 'LogBSP', 'BSP', 'Year', 'HasPrice5Min', 'Price5Min']
FEATURE_COLS = [c for c in train_live.columns if c not in exclude]

print(f"  Train Live: {len(train_live):,} | Test Live: {len(test_live):,}")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Feature List: {FEATURE_COLS}")

print("[3/5] Training AutoGluon V20...")
model_path = f'models/autogluon_bsp_v20_{random.randint(1000,9999)}'
predictor = TabularPredictor(
    label='LogBSP',
    eval_metric='mean_absolute_error',
    path=model_path,
    problem_type='regression'
).fit(
    train_data=train_live[FEATURE_COLS + ['LogBSP']],
    time_limit=300,  # 5 min
    presets='best_quality'
)

print("[4/5] Evaluating...")
preds_log = predictor.predict(test_live[FEATURE_COLS])
test_live['PredLogBSP'] = preds_log
test_live['PredBSP'] = np.exp(preds_log)

# Metrics
mae_log = np.mean(np.abs(test_live['LogBSP'] - test_live['PredLogBSP']))
mae_price = np.mean(np.abs(test_live['BSP'] - test_live['PredBSP']))
mape = np.mean(np.abs(test_live['BSP'] - test_live['PredBSP']) / test_live['BSP']) * 100

print("\n" + "="*70)
print("V20 RESULTS (Top 30 + Price5Min)")
print("="*70)
print(f"Log MAE:   {mae_log:.4f}")
print(f"Price MAE: ${mae_price:.2f}")
print(f"MAPE:      {mape:.1f}%")

print("\nAccuracy by Price Range:")
for low, high in [(1, 3), (3, 5), (5, 10), (10, 20), (20, 50)]:
    subset = test_live[(test_live['BSP'] >= low) & (test_live['BSP'] < high)]
    if len(subset) > 0:
        sub_mape = np.mean(np.abs(subset['BSP'] - subset['PredBSP']) / subset['BSP']) * 100
        print(f"${low}-${high}: MAPE {sub_mape:.1f}% ({len(subset):,} bets)")

print("-" * 30)
try:
    fi = predictor.feature_importance(test_live[FEATURE_COLS + ['LogBSP']].sample(5000))
    print(fi.head(10))
except:
    pass

print(f"\nModel saved to: {model_path}")
print("="*70)
