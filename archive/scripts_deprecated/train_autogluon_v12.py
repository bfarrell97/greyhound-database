"""
AutoGluon with V12 Feature Set - No Odds Restriction
Tests natural model performance across ALL price ranges
Uses 69 top features from Feature Factory V1
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
import time
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

print("="*70)
print("AUTOGLUON V12 FEATURES - NO ODDS RESTRICTION")
print("Train: 2020-2023 | Test: 2024 + 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
       ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID, g.DamID, g.DateOfBirth
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = (df['Position'] == 1).astype(int)
df['Tier'] = df['TrackName'].apply(get_tier)

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
            'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']

print(f"[1/5] Loaded {len(df):,} entries")

print("[2/5] Building V12 feature set (69 features)...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
trainer_recent = defaultdict(list)
trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))

train_rows = []
test_24_rows = []
test_25_rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 4: continue
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    track_id = race_df['TrackID'].iloc[0]
    tier = race_df['Tier'].iloc[0]
    year = race_date.year
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_history.get(dog_id, [])
        
        if len(hist) >= 3:
            recent = hist[-10:]
            times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
            beyers = [h['beyer'] for h in recent if h['beyer'] is not None]
            positions = [h['position'] for h in recent if h['position'] is not None]
            closings = [h['closing'] for h in recent if h['closing'] is not None]
            deltas = [h['pos_delta'] for h in recent if h['pos_delta'] is not None]
            weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
            splits = [h['split'] for h in recent if h['split'] is not None]
            
            if len(times) >= 3:
                features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Distance': distance, 'MeetingDate': race_date}
                
                # ============ Time Features ============
                features['TimeBest'] = min(times)
                features['TimeWorst'] = max(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                features['TimeStd'] = np.std(times)
                features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                features['TimeBestRecent3'] = min(times[-3:])
                features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                
                # ============ Split Features ============
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                features['SplitLag1'] = splits[-1] if splits else 0
                features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                
                # ============ Speed Figures ============
                features['BeyerLag1'] = beyers[-1] if beyers else 77
                features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                
                # ============ Career Stats ============
                features['PosAvg'] = np.mean(positions)
                features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                features['CareerStarts'] = min(len(hist), 100)
                features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                
                features['LastWonDaysAgo'] = 999
                for h in reversed(hist):
                    if h['position'] == 1:
                        features['LastWonDaysAgo'] = (race_date - h['date']).days
                        break
                
                # ============ Trainer Features ============
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                cutoff_30d = race_date - timedelta(days=30)
                cutoff_60d = race_date - timedelta(days=60)
                t_rec = trainer_recent.get(trainer_id, [])
                rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                rec_60 = [x for x in t_rec if x[0] >= cutoff_60d]
                features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                features['TrainerWinRate60d'] = safe_div(sum(x[1] for x in rec_60), len(rec_60), features['TrainerWinRate']) if len(rec_60) >= 10 else features['TrainerWinRate']
                features['TrainerStarts60d'] = len(rec_60)
                features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                
                t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                features['TrainerTrackRuns'] = t_track['runs']
                features['Trainer_x_Track'] = features['TrainerWinRate'] * features['TrainerTrackRuns']
                
                dist_key = round(distance / 100) * 100
                t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                
                # ============ Distance/Track Specialization ============
                dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                features['DistExperience'] = min(len(dist_runs), 30) / 30
                features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                
                track_runs = [h for h in hist if h['track_id'] == track_id]
                features['TrackExperience'] = min(len(track_runs), 20) / 20
                features['TrackAvgPos'] = np.mean([t['position'] for t in track_runs]) if track_runs else features['PosAvg']
                features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), 0.35) if len(track_runs) >= 3 else 0.35
                
                tier_runs = [h for h in hist if h.get('tier', 0) == tier]
                features['TierExperience'] = min(len(tier_runs), 30) / 30
                
                # ============ Box Features ============
                box = int(r['Box']) if pd.notna(r['Box']) else 4
                features['Box'] = box
                
                tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                
                inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                outside_runs = [h for h in hist if h.get('box', 4) > 4]
                inside_rate = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['CareerWinRate']) if len(inside_runs) >= 3 else features['CareerWinRate']
                outside_rate = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['CareerWinRate']) if len(outside_runs) >= 3 else features['CareerWinRate']
                features['BoxPreference'] = inside_rate - outside_rate
                
                this_box_runs = [h for h in hist if h.get('box', 0) == box]
                features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['CareerWinRate']) if len(this_box_runs) >= 2 else features['CareerWinRate']
                
                # ============ Age Features ============
                age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                features['Age_x_Experience'] = age_months * features['CareerStarts']
                
                # ============ Bloodline Features ============
                sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                features['DamRuns'] = dam_data['runs']
                features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                features['BloodlineVsDog'] = features['BloodlineScore'] - features['CareerWinRate']
                features['Bloodline_x_Age'] = features['BloodlineScore'] * age_months
                
                # ============ Weight Features ============
                weight_avg = np.mean(weights) if weights else 30
                current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                features['WeightAvg'] = weight_avg
                features['WeightChange'] = current_weight - weight_avg
                features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                features['Weight_x_Distance'] = current_weight * distance / 1000
                
                # ============ Rest/Frequency ============
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                
                # ============ Pace Features ============
                deltas_array = deltas if deltas else [0]
                features['PosImprovement'] = np.mean(deltas_array)
                
                # ============ Interaction Features ============
                features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                features['Time_x_TrainerForm'] = features['TimeBest'] * features['TrainerFormVsAll']
                features['Form_x_Trainer'] = (sum(1 for p in positions[-3:] if p <= 3) / 3 if len(positions) >= 3 else 0.5) * features['TrainerWinRate']
                features['Rest_x_Form'] = days_since * features['WinRate5']
                features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrainerTrackWinRate']) / 3
                
                if year <= 2023:
                    train_rows.append(features)
                elif year == 2024:
                    test_24_rows.append(features)
                else:
                    test_25_rows.append(features)
    
    # Update history
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        box = int(r['Box']) if pd.notna(r['Box']) else 4
        won = r['Won']
        
        dog_history[dog_id].append({
            'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
            'position': r['Position'], 'track_id': track_id, 'distance': distance, 'tier': tier,
            'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
            'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
            'weight': r['Weight'], 'box': box, 'split': r['Split'] if pd.notna(r['Split']) else None
        })
        
        if pd.notna(r['TrainerID']):
            tid = r['TrainerID']
            trainer_all[tid]['runs'] += 1
            if won: trainer_all[tid]['wins'] += 1
            trainer_recent[tid].append((race_date, won, track_id, distance))
            cutoff = race_date - timedelta(days=120)
            trainer_recent[tid] = [x for x in trainer_recent[tid] if x[0] >= cutoff]
            trainer_track[tid][track_id]['runs'] += 1
            if won: trainer_track[tid][track_id]['wins'] += 1
            dist_key = round(distance / 100) * 100
            trainer_dist[tid][dist_key]['runs'] += 1
            if won: trainer_dist[tid][dist_key]['wins'] += 1
        
        if pd.notna(r['SireID']):
            sire_stats[r['SireID']]['runs'] += 1
            if won: sire_stats[r['SireID']]['wins'] += 1
        if pd.notna(r['DamID']):
            dam_stats[r['DamID']]['runs'] += 1
            if won: dam_stats[r['DamID']]['wins'] += 1
        
        track_box_wins[track_id][box]['runs'] += 1
        if won: track_box_wins[track_id][box]['wins'] += 1
    
    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

train_df = pd.DataFrame(train_rows)
test_24_df = pd.DataFrame(test_24_rows)
test_25_df = pd.DataFrame(test_25_rows)

exclude_cols = ['RaceID', 'Won', 'BSP', 'Distance', 'MeetingDate']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} | Test 2024: {len(test_24_df):,} | Test 2025: {len(test_25_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

# Prepare data
train_ag = train_df[FEATURE_COLS + ['Won']].copy()
test_24_ag = test_24_df[FEATURE_COLS].copy()
test_25_ag = test_25_df[FEATURE_COLS].copy()

import random
model_path = f'models/autogluon_v12_features_{random.randint(1000,9999)}'
print(f"[3/5] Training AutoGluon (5 min time limit) to {model_path}...")
predictor = TabularPredictor(
    label='Won',
    eval_metric='roc_auc',
    path=model_path,
    verbosity=1  # Less verbose
).fit(
    train_data=train_ag,
    time_limit=300,
    presets='best_quality'
)

print("\n[4/5] Testing on 2024 and 2025...")

# 2024 predictions
test_24_df['PredProb'] = predictor.predict_proba(test_24_ag)[1]
test_24_df['MarginOverSecond'] = test_24_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
leaders_24 = test_24_df.loc[test_24_df.groupby('RaceID')['PredProb'].idxmax()]

# 2025 predictions
test_25_df['PredProb'] = predictor.predict_proba(test_25_ag)[1]
test_25_df['MarginOverSecond'] = test_25_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)
leaders_25 = test_25_df.loc[test_25_df.groupby('RaceID')['PredProb'].idxmax()]

def test_strategy(leaders, name, conf=0, price_low=0, price_high=999):
    if price_low > 0:
        strat = leaders[(leaders['MarginOverSecond'] >= conf) & 
                        (leaders['BSP'] >= price_low) & (leaders['BSP'] <= price_high)]
    else:
        strat = leaders[leaders['MarginOverSecond'] >= conf]
    
    valid = strat.dropna(subset=['BSP'])
    if len(valid) < 30:
        return None
    wins = valid['Won'].sum()
    sr = wins / len(valid) * 100
    profit = (valid[valid['Won']==1]['BSP'].sum() * 0.9) - len(valid)
    roi = profit / len(valid) * 100
    return {'name': name, 'bets': len(valid), 'wins': wins, 'sr': sr, 'roi': roi}

print("\n" + "="*70)
print("AUTOGLUON V12 RESULTS - NO PRICE RESTRICTION")
print("="*70)

print("\n2024 Results by Confidence (ALL PRICES):")
for conf in [0.0, 0.05, 0.08, 0.10, 0.12]:
    r = test_strategy(leaders_24, f"Conf {conf:.0%}", conf)
    if r:
        print(f"  {r['name']}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}%")

print("\n2025 Results by Confidence (ALL PRICES - OUT-OF-SAMPLE):")
for conf in [0.0, 0.05, 0.08, 0.10, 0.12]:
    r = test_strategy(leaders_25, f"Conf {conf:.0%}", conf)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  {r['name']}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

print("\n2025 Results by Price Range (10% confidence):")
for low, high in [(1, 3), (3, 8), (8, 20), (20, 100)]:
    r = test_strategy(leaders_25, f"${low}-${high}", 0.10, low, high)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  {r['name']}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")
print("="*70)
