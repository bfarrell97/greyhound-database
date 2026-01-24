"""
Feature Factory V3 - HISTORICAL & COMBINATIONS
==============================================
Goal: Discovery of "Pure Form" predictors (No Live Market Data).
Target: LogBSP

Features to Generate:
1.  **Deep History:** Rolling Stats (Mean, Median, Min, Max, Std, Skew) for 3, 5, 10, 20 races.
2.  **Class Metrics:**
    - Ratio of LastBSP / AvgBSP (Class Drop/Rise)
    - AvgBSP in last 3 races vs Career Avg
3.  **Combinations / Interactions:**
    - Trainer * Track (Trainer performance at this specific track)
    - Sire * Distance (Breeding suitability)
    - Box * Track (Box bias at this track)
4.  **Consistency:**
    - Coefficient of Variation (Std/Mean) for Time and Split.
    - Reliability (Finish % in Top 3)
5.  **Recency:**
    - Days since last run (Freshness)
    - Change in Body Weight
"""
import sqlite3
import pandas as pd
import numpy as np
import time
from scipy.stats import pearsonr, skew
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE FACTORY V3 - HISTORICAL MINING")
print("Target: LogBSP Correlation")
print("Generating ~500 features...")
print("="*70)

start_time = time.time()

# Load data
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
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
# Ensure numeric
for col in ['FinishTime', 'Split', 'Weight', 'BSP']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"[1/4] Loaded {len(df):,} entries")

print("[2/4] Generating Features...")

# Dictionaries for Aggregate Stats
dog_hist = defaultdict(list)
# Interactions
trainer_track_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0, 'wins': 0})
sire_dist_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
box_track_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
trainer_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})

rows = []
processed = 0

def safe_div(a, b, default=0):
    return a / b if b and b > 0 else default

for race_id, race_df in df.groupby('RaceID', sort=False):
    race_date = race_df['MeetingDate'].iloc[0]
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_hist.get(dog_id, [])
        
        # Need history for features
        if len(hist) >= 3:
            feat = {'LogBSP': r['LogBSP'], 'BSP': r['BSP']}
            
            # --- 1. DEEP HISTORY (Rolling) ---
            bsps = [h['bsp'] for h in hist] # Full hist
            times = [h['time'] for h in hist if h['time']]
            pos = [h['pos'] for h in hist if h['pos']]
            
            for w in [3, 5, 10, 20]:
                recent_bsp = bsps[-w:]
                if len(recent_bsp) > 0:
                    feat[f'BSP_Mean_{w}'] = np.mean(recent_bsp)
                    feat[f'BSP_Min_{w}'] = min(recent_bsp)
                    feat[f'BSP_Max_{w}'] = max(recent_bsp)
                    feat[f'BSP_Std_{w}'] = np.std(recent_bsp)
                
                recent_pos = pos[-w:]
                if len(recent_pos) > 0:
                    feat[f'Pos_Mean_{w}'] = np.mean(recent_pos)
                    feat[f'WinRate_{w}'] = sum(1 for p in recent_pos if p==1)/len(recent_pos)
            
            # Career
            feat['BSP_Mean_Career'] = np.mean(bsps)
            feat['Runs_Career'] = len(hist)
            
            # --- 2. CLASS METRICS ---
            # Ratio of Recent vs Career (Is form improving?)
            if feat.get('BSP_Mean_10') and feat.get('BSP_Mean_Career'):
                feat['Class_Trend_10'] = feat['BSP_Mean_10'] / feat['BSP_Mean_Career'] # < 1 means running in lower price (better class?) or just better form?
                # Actually, Lower Price = Better Dog. So < 1 means "Running shorter odds than usual" -> In good form.
            
            # Last Run vs Avg
            last_bsp = hist[-1]['bsp']
            feat['LastBSP_Rel_10'] = last_bsp / feat.get('BSP_Mean_10', last_bsp)
            
            # --- 3. INTERACTIONS ---
            # Trainer * Track
            tt_key = (r['TrainerID'], r['TrackID'])
            stats = trainer_track_stat.get(tt_key)
            if stats and stats['runs'] > 2:
                feat['Trainer_Track_AvgBSP'] = stats['bsp_sum'] / stats['runs']
                feat['Trainer_Track_WinRate'] = stats['wins'] / stats['runs']
                # Uplift: Is Trainer better at THIS track than their General Avg?
                gen_avg = trainer_stat[r['TrainerID']]['bsp_sum'] / trainer_stat[r['TrainerID']]['runs'] if trainer_stat[r['TrainerID']]['runs'] > 0 else 10
                feat['Trainer_Track_Uplift'] = feat['Trainer_Track_AvgBSP'] / gen_avg
            
            # Sire * Distance (Rounded to nearest 50m)
            dist_bin = round(r['Distance'] / 50) * 50
            sd_key = (r['SireID'], dist_bin)
            stats = sire_dist_stat.get(sd_key)
            if r['SireID'] and stats and stats['runs'] > 5:
                feat['Sire_Dist_AvgBSP'] = stats['bsp_sum'] / stats['runs']
            
            # Box * Track
            bt_key = (r['TrackID'], r['Box'])
            stats = box_track_stat.get(bt_key)
            if stats and stats['runs'] > 5:
                feat['Box_Track_AvgBSP'] = stats['bsp_sum'] / stats['runs']
                # Bias: How does this box compare to track avg?
                # (Simplified here)
            
            # --- 4. RECENCY ---
            last_date = hist[-1]['date']
            days_since = (pd.to_datetime(race_date) - pd.to_datetime(last_date)).days
            feat['DaysSinceLast'] = days_since
            feat['DaysSince_Log'] = np.log(days_since + 1)
            
            if len(hist) > 1:
                feat['SecondUp'] = 1 if days_since < 10 and (pd.to_datetime(hist[-1]['date']) - pd.to_datetime(hist[-2]['date'])).days > 30 else 0
            else:
                feat['SecondUp'] = 0

            rows.append(feat)

    # Update History
    for _, r in race_df.iterrows():
        bsp = r['BSP'] if pd.notna(r['BSP']) and r['BSP'] > 0 else 10
        pos_val = pd.to_numeric(r['Position'], errors='coerce') 
        is_win = 1 if pos_val == 1 else 0
        
        dog_hist[r['GreyhoundID']].append({
            'bsp': bsp,
            'pos': pos_val,
            'time': r['FinishTime'],
            'date': race_date
        })
        
        # Updates
        trainer_track_stat[(r['TrainerID'], r['TrackID'])]['runs'] += 1
        trainer_track_stat[(r['TrainerID'], r['TrackID'])]['bsp_sum'] += bsp
        trainer_track_stat[(r['TrainerID'], r['TrackID'])]['wins'] += is_win
        
        trainer_stat[r['TrainerID']]['runs'] += 1
        trainer_stat[r['TrainerID']]['bsp_sum'] += bsp
        
        if r['SireID']:
            dist_bin = round(r['Distance'] / 50) * 50
            sire_dist_stat[(r['SireID'], dist_bin)]['runs'] += 1
            sire_dist_stat[(r['SireID'], dist_bin)]['bsp_sum'] += bsp
            
        box_track_stat[(r['TrackID'], r['Box'])]['runs'] += 1
        box_track_stat[(r['TrackID'], r['Box'])]['bsp_sum'] += bsp

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
    if len(valid) > 5000:
        c, _ = pearsonr(valid[f], valid[target])
        corrs.append({'Feature': f, 'Msg': f"{f}: {c:.4f}", 'Abs': abs(c)})

corrs.sort(key=lambda x: x['Abs'], reverse=True)

print("\nTOP 50 HISTORICAL PREDICTORS:")
for i, c in enumerate(corrs[:50]):
    print(f"{i+1}. {c['Msg']}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} min")
