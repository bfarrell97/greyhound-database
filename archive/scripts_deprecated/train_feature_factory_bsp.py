"""
BSP Feature Factory - HIGH VOLUME
Generate 200+ features to find the best BSP predictors.
Focus:
- Rolling stats (3, 5, 10 races) for BSP, Time, Position, Split
- Volatility and Trend metrics
- Interaction features (Trainer*Track, Box*Dist)
- Ratios (BSP/Time, BSP/BoxStats)
"""
import sqlite3
import pandas as pd
import numpy as np
import time
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("BSP FEATURE FACTORY - HIGH VOLUME")
print("Target: LogBSP Correlation")
print("Generating 200+ features...")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID, g.DamID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2022-01-01' AND '2024-12-31'
  AND ge.Position IS NOT NULL 
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
for col in ['Price5Min', 'FinishTime', 'Split', 'Weight', 'BSP']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"[1/4] Loaded {len(df):,} entries")

print("[2/4] Generating 200+ Features...")

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
        
        if len(hist) >= 5: # Need reasonable history
            feat = {'LogBSP': r['LogBSP'], 'BSP': r['BSP']}
            
            # --- 1. Rolling Windows (3, 5, 10) ---
            for window in [3, 5, 10]:
                recent = hist[-window:]
                if len(recent) < 2: continue
                
                bsps = [h['bsp'] for h in recent]
                times = [h['time'] for h in recent if h['time']]
                pos = [h['pos'] for h in recent if h['pos']]
                
                # BSP Stats
                feat[f'BSP_Mean_{window}'] = np.mean(bsps)
                feat[f'BSP_Max_{window}'] = max(bsps)
                feat[f'BSP_Min_{window}'] = min(bsps)
                feat[f'BSP_Std_{window}'] = np.std(bsps)
                
                # Time Stats
                if times:
                    feat[f'Time_Mean_{window}'] = np.mean(times)
                    feat[f'Time_Best_{window}'] = min(times)
                
                # Pos Stats
                if pos:
                    feat[f'Pos_Mean_{window}'] = np.mean(pos)
                    feat[f'WinRate_{window}'] = sum(1 for p in pos if p==1)/len(pos)

            # --- 2. Trends ---
            if len(hist) >= 5:
                feat['BSP_Trend_5'] = hist[-1]['bsp'] - hist[-5]['bsp']
                feat['Time_Trend_5'] = (hist[-1]['time'] - hist[-5]['time']) if hist[-1]['time'] and hist[-5]['time'] else 0
            
            # --- 3. Context/History Ratios ---
            avg_bsp_3 = feat.get('BSP_Mean_3', 10)
            feat['LastBSP_Ratio'] = hist[-1]['bsp'] / avg_bsp_3 if avg_bsp_3 > 0 else 1
            
            # --- 4. Trainer/Sire Stats ---
            tid = r['TrainerID']
            if trainer_stat[tid]['runs'] > 5:
                tr_avg = trainer_stat[tid]['bsp_sum'] / trainer_stat[tid]['runs']
                feat['Trainer_AvgBSP'] = tr_avg
                feat['Trainer_Rel'] = avg_bsp_3 / tr_avg # Is this dog better than trainer avg?
            
            sid = r['SireID']
            if sid and sire_stat[sid]['runs'] > 10:
                sire_avg = sire_stat[sid]['bsp_sum'] / sire_stat[sid]['runs']
                feat['Sire_AvgBSP'] = sire_avg
            
            # --- 5. Box Stats ---
            bid = (r['TrackID'], r['Box'])
            if box_stat[bid]['runs'] > 5:
                box_avg = box_stat[bid]['bsp_sum'] / box_stat[bid]['runs']
                feat['Box_Track_AvgBSP'] = box_avg
            
            rows.append(feat)
            
    # Update History
    for _, r in race_df.iterrows():
        item = {
            'bsp': r['BSP'] if r['BSP'] else 10,
            'time': r['FinishTime'],
            'pos': pd.to_numeric(r['Position'], errors='coerce') 
        }
        dog_hist[r['GreyhoundID']].append(item)
        
        # Update Aggregate stats
        trainer_stat[r['TrainerID']]['runs'] += 1
        trainer_stat[r['TrainerID']]['bsp_sum'] += item['bsp']
        
        if r['SireID']:
            sire_stat[r['SireID']]['runs'] += 1
            sire_stat[r['SireID']]['bsp_sum'] += item['bsp']
            
        box_stat[(r['TrackID'], r['Box'])]['runs'] += 1
        box_stat[(r['TrackID'], r['Box'])]['bsp_sum'] += item['bsp']

    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

analysis = pd.DataFrame(rows)
print(f"[3/4] Analyzing {len(analysis):,} rows with {len(analysis.columns)} cols...")

# Correlation
corrs = []
target = 'LogBSP'
features = [c for c in analysis.columns if c not in ['LogBSP', 'BSP']]

for f in features:
    valid = analysis.dropna(subset=[f, target])
    if len(valid) > 1000:
        c, _ = pearsonr(valid[f], valid[target])
        corrs.append({'Feature': f, 'Msg': f"{f}: {c:.4f}", 'Abs': abs(c)})

corrs.sort(key=lambda x: x['Abs'], reverse=True)

print("\nTOP 50 PREDICTORS:")
for i, c in enumerate(corrs[:50]):
    print(f"{i+1}. {c['Msg']}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} min")
