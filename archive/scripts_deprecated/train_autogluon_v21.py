"""
AutoGluon V21 - PURE FORM / HISTORY MODEL
Target: LogBSP
Features: Historical Only (No Live Market Data)
Key Features: BSP_Mean_10 (Class), Trainer_AvgBSP, Pos_Mean_10
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
print("AUTOGLUON V21 - PURE FORM MODEL")
print("Features: History Only (Class, Trainer, Box, Position)")
print("Target: LogBSP")
print("="*70)

start_time = time.time()

# Load data (Same query, but we won't use Price5Min for features)
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID, g.DamID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
  AND ge.Position IS NOT NULL 
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Year'] = df['MeetingDate'].dt.year
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))

print(f"[1/5] Loaded {len(df):,} entries")

print("[2/5] Building Historical Features...")

from collections import defaultdict
dog_hist = defaultdict(list)
trainer_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
sire_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
# dam_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0}) # Maybe too sparse?
box_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})

rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_hist.get(dog_id, [])
        
        # Need history for "Pure Form" prediction
        if len(hist) >= 3:
            feat = {
                'RaceID': race_id,
                'LogBSP': r['LogBSP'], 
                'BSP': r['BSP'],
                'Year': r['Year'],
                'Box': int(r['Box']) if r['Box'] else 0,
                'Distance': r['Distance']
            }
            
            # --- HISTORICAL CLASS FEATURES (The Core of V21) ---
            
            # 1. BSP Rolling Stats (The best proxy for "Class")
            bsps = [h['bsp'] for h in hist[-10:]]
            if len(bsps) >= 3:
                feat['BSP_Mean_10'] = np.mean(bsps)
                feat['BSP_Max_10'] = max(bsps)
                feat['BSP_Min_10'] = min(bsps)
                feat['BSP_Std_10'] = np.std(bsps)
                
                feat['BSP_Mean_3'] = np.mean(bsps[-3:])
                feat['BSP_Min_3'] = min(bsps[-3:]) # Recent form peak
            
            # 2. Position/Win Stats
            pos = [h['pos'] for h in hist[-10:] if h['pos']]
            if len(pos) >= 3:
                feat['Pos_Mean_10'] = np.mean(pos)
                feat['WinRate_10'] = sum(1 for p in pos if p==1)/len(pos)
                feat['PlaceRate_10'] = sum(1 for p in pos if p<=3)/len(pos)
            else:
                feat['Pos_Mean_10'] = 5
                feat['WinRate_10'] = 0
                feat['PlaceRate_10'] = 0

            # 3. Contextual Stats (Trainer/Sire/Box)
            # Trainer
            tid = r['TrainerID']
            if trainer_stat[tid]['runs'] > 5:
                # Is this trainer usually betting favorites?
                feat['Trainer_AvgBSP'] = trainer_stat[tid]['bsp_sum'] / trainer_stat[tid]['runs']
            else:
                feat['Trainer_AvgBSP'] = 10 
            
            # Sire
            sid = r['SireID']
            if sid and sire_stat[sid]['runs'] > 10:
                feat['Sire_AvgBSP'] = sire_stat[sid]['bsp_sum'] / sire_stat[sid]['runs']
            else:
                feat['Sire_AvgBSP'] = 10
            
            # Box @ Track
            bid = (r['TrackID'], r['Box'])
            if box_stat[bid]['runs'] > 5:
                feat['Box_Track_AvgBSP'] = box_stat[bid]['bsp_sum'] / box_stat[bid]['runs']
            else:
                feat['Box_Track_AvgBSP'] = 10
            
            # 4. Trends
            if len(hist) >= 5:
                feat['BSP_Trend_5'] = hist[-1]['bsp'] - hist[-5]['bsp'] # Are they getting seemingly worse/better?
            else:
                feat['BSP_Trend_5'] = 0
                
            # 5. Last Run Delta
            feat['LastBSP'] = hist[-1]['bsp']
            feat['LastPos'] = hist[-1]['pos']
            
            rows.append(feat)

    # Update History
    for _, r in race_df.iterrows():
        bsp_val = r['BSP'] if r['BSP'] else 10
        pos_val = pd.to_numeric(r['Position'], errors='coerce')
        
        dog_hist[r['GreyhoundID']].append({
            'bsp': bsp_val,
            'pos': pos_val
        })
        
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
train_df = full_df[full_df['Year'] <= 2023].copy()
test_df = full_df[full_df['Year'] >= 2024].copy()

exclude = ['RaceID', 'LogBSP', 'BSP', 'Year']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude]

print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Feature List: {FEATURE_COLS}")

print("[3/5] Training AutoGluon V21...")
model_path = f'models/autogluon_bsp_v21_{random.randint(1000,9999)}'
predictor = TabularPredictor(
    label='LogBSP',
    eval_metric='mean_absolute_error',
    path=model_path,
    problem_type='regression'
).fit(
    train_data=train_df[FEATURE_COLS + ['LogBSP']],
    time_limit=300, 
    presets='best_quality'
)

print("[4/5] Evaluating...")
preds_log = predictor.predict(test_df[FEATURE_COLS])
test_df['PredLogBSP'] = preds_log
test_df['PredBSP'] = np.exp(preds_log)

# Metrics
mae_log = np.mean(np.abs(test_df['LogBSP'] - test_df['PredLogBSP']))
mae_price = np.mean(np.abs(test_df['BSP'] - test_df['PredBSP']))
mape = np.mean(np.abs(test_df['BSP'] - test_df['PredBSP']) / test_df['BSP']) * 100

print("\n" + "="*70)
print("V21 RESULTS (PURE FORM)")
print("="*70)
print(f"Log MAE:   {mae_log:.4f}")
print(f"Price MAE: ${mae_price:.2f}")
print(f"MAPE:      {mape:.1f}%")

print("\nAccuracy by Price Range:")
for low, high in [(1, 3), (3, 5), (5, 10), (10, 20), (20, 50)]:
    subset = test_df[(test_df['BSP'] >= low) & (test_df['BSP'] < high)]
    if len(subset) > 0:
        sub_mape = np.mean(np.abs(subset['BSP'] - subset['PredBSP']) / subset['BSP']) * 100
        print(f"${low}-${high}: MAPE {sub_mape:.1f}% ({len(subset):,} bets)")

print("-" * 30)
try:
    fi = predictor.feature_importance(test_df[FEATURE_COLS + ['LogBSP']].sample(5000))
    print(fi.head(10))
except:
    pass

print(f"\nModel saved to: {model_path}")
print("="*70)
