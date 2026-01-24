"""
V12 Hybrid Strategy Backtesting
Testing specific strategy combinations
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def backtest(df, label, min_bets=30):
    if len(df) < min_bets: 
        return None
    wins = df['Won'].sum()
    sr = wins / len(df) * 100
    valid = df.dropna(subset=['BSP'])
    if len(valid) == 0: return None
    returns = valid[valid['Won'] == 1]['BSP'].sum()
    profit = returns - len(valid)
    roi = profit / len(valid) * 100
    print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%, Profit: {profit:+.1f}")
    return {'Label': label, 'Bets': len(df), 'Wins': wins, 'SR': sr, 'ROI': roi, 'Profit': profit}

print("="*70)
print("V12 HYBRID STRATEGY BACKTESTING")
print("="*70)

# Load model
print("\n[1/4] Loading V12 model...")
with open('models/pace_v13_optimized.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['features']

# Load data
print("[2/4] Loading data...")
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

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
WHERE rm.MeetingDate BETWEEN '2024-06-01' AND '2025-11-30'
  AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = (df['Position'] == 1).astype(int)
df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
            'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Tier'] = df['TrackName'].apply(get_tier)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']

print(f"  Loaded {len(df):,} entries")

# Build features (abbreviated - same as other scripts)
print("[3/4] Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
trainer_recent = defaultdict(list)
trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))

feature_rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 4: continue
    
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    track_id = race_df['TrackID'].iloc[0]
    track_name = race_df['TrackName'].iloc[0]
    tier = race_df['Tier'].iloc[0]
    day_of_week = race_df['DayOfWeek'].iloc[0]
    
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
                features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Tier': tier, 
                           'Distance': distance, 'TrackName': track_name, 'DayOfWeek': day_of_week,
                           'Box': int(r['Box'])}
                
                # Core features (same as backtest scripts)
                features['TimeBest'] = min(times)
                features['TimeWorst'] = max(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                features['TimeStd'] = np.std(times) if len(times) >= 3 else 0
                features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                features['TimeBestRecent3'] = min(times[-3:])
                features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                features['SplitLag1'] = splits[-1] if splits else 0
                features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                
                features['BeyerLag1'] = beyers[-1] if beyers else 77
                features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                
                features['PosAvg'] = np.mean(positions)
                features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                features['CareerStarts'] = min(len(hist), 100)
                features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                features['LastWonDaysAgo'] = 999
                for i, h in enumerate(reversed(hist)):
                    if h['position'] == 1:
                        features['LastWonDaysAgo'] = (race_date - h['date']).days
                        break
                
                form_trend = 0
                if len(positions) >= 5:
                    first_half = np.mean(positions[:len(positions)//2])
                    second_half = np.mean(positions[len(positions)//2:])
                    form_trend = first_half - second_half
                features['FormTrend'] = form_trend
                
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
                features['TrainerTrackRuns'] = min(t_track['runs'], 100) / 100
                
                dist_key = round(distance / 100) * 100
                t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                
                dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                features['DistExperience'] = min(len(dist_runs), 30) / 30
                features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                
                track_runs = [h for h in hist if h['track_id'] == track_id]
                features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), 0.35) if len(track_runs) >= 3 else 0.35
                features['TrackExperience'] = min(len(track_runs), 20) / 20
                features['TrackAvgPos'] = np.mean([t['position'] for t in track_runs]) if track_runs else features['PosAvg']
                
                tier_runs = [h for h in hist if h.get('tier', 0) == tier]
                features['TierExperience'] = min(len(tier_runs), 30) / 30
                
                box = int(r['Box'])
                tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                
                age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                features['AgeMonths'] = age_months
                features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                features['AgePeakDist'] = abs(age_months - 30)
                
                deltas_array = deltas if deltas else [0]
                features['PosImprovement'] = np.mean(deltas_array)
                features['ClosingAvg'] = np.mean(closings) if closings else 0
                features['ClosingBest'] = min(closings) if closings else 0
                features['DeltaBest'] = max(deltas) if deltas else 0
                
                sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                features['DamRuns'] = min(dam_data['runs'], 200) / 200
                features['SireRuns'] = min(sire_data['runs'], 500) / 500
                features['BloodlineVsDog'] = features['BloodlineScore'] - features['CareerWinRate']
                
                weight_avg = np.mean(weights) if weights else 30
                current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                features['Weight'] = current_weight
                features['WeightAvg'] = weight_avg
                features['WeightChange'] = current_weight - weight_avg
                features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                
                inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                outside_runs = [h for h in hist if h.get('box', 4) > 4]
                inside_rate = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['CareerWinRate']) if len(inside_runs) >= 3 else features['CareerWinRate']
                outside_rate = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['CareerWinRate']) if len(outside_runs) >= 3 else features['CareerWinRate']
                features['BoxPreference'] = inside_rate - outside_rate
                
                this_box_runs = [h for h in hist if h.get('box', 0) == box]
                features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['CareerWinRate']) if len(this_box_runs) >= 2 else features['CareerWinRate']
                
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                rest_score = max(0, 1 - abs(days_since - 10) / 20)
                features['RestScore'] = rest_score
                
                # Derived
                features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                features['Time_x_TrainerForm'] = features['TimeBest'] * features['TrainerWinRate30d']
                features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                features['Trainer_x_Track'] = features['TrainerWinRate'] * features['TrainerTrackWinRate']
                features['Age_x_Experience'] = (age_months / 48) * (features['CareerStarts'] / 100)
                features['Form_x_Trainer'] = form_trend * features['TrainerWinRate']
                features['Weight_x_Distance'] = features['WeightChange'] * (1 if distance > 500 else -1)
                features['Bloodline_x_Age'] = features['BloodlineScore'] * (1 - features['AgePeakDist'] / 20)
                features['Rest_x_Form'] = rest_score * form_trend
                features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrainerTrackWinRate']) / 3
                
                feature_rows.append(features)
    
    # Update lookups
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
    if processed % 10000 == 0:
        print(f"  {processed:,} races...")

print(f"  Total: {processed:,} races")

# Predict
print("[4/4] Running predictions...")
feat_df = pd.DataFrame(feature_rows)
feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')

for col in feature_cols:
    if col not in feat_df.columns:
        feat_df[col] = 0

X = feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_scaled = scaler.transform(X)
feat_df['PredProb'] = model.predict_proba(X_scaled)[:, 1]

# Calculate margin over second
feat_df['MarginOverSecond'] = feat_df.groupby('RaceID')['PredProb'].transform(lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0)

# Race leaders
race_leaders = feat_df.loc[feat_df.groupby('RaceID')['PredProb'].idxmax()].copy()

print(f"  Generated {len(race_leaders):,} predictions\n")

# ==============================================================
# HYBRID STRATEGY TESTS
# ==============================================================
print("="*70)
print("HYBRID STRATEGY RESULTS")
print("="*70)

# Strategy 1: High Confidence + $3-$8 + Short Distance (<550m)
print("\n--- STRATEGY 1: HIGH CONFIDENCE + $3-$8 + DIST < 550m ---")
s1 = race_leaders[
    (race_leaders['MarginOverSecond'] >= 0.10) & 
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8) &
    (race_leaders['Distance'] < 550)
]
backtest(s1, "High Conf + $3-$8 + <550m")

# Variants
s1a = race_leaders[
    (race_leaders['MarginOverSecond'] >= 0.12) & 
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8) &
    (race_leaders['Distance'] < 550)
]
backtest(s1a, "Higher Conf (12%) + $3-$8 + <550m")

s1b = race_leaders[
    (race_leaders['MarginOverSecond'] >= 0.10) & 
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 6) &
    (race_leaders['Distance'] < 550)
]
backtest(s1b, "High Conf + $3-$6 + <550m")

# Strategy 2: Long Shots with filters
print("\n--- STRATEGY 2: LONG SHOTS ($10+) ---")

# Long shots at good tracks
s2a = race_leaders[
    (race_leaders['BSP'] >= 10) & (race_leaders['BSP'] <= 20) &
    (race_leaders['Tier'] >= 1)  # Metro or Provincial only
]
backtest(s2a, "Longshot $10-$20 + Metro/Provincial")

# Long shots with high confidence
s2b = race_leaders[
    (race_leaders['BSP'] >= 10) & (race_leaders['BSP'] <= 20) &
    (race_leaders['MarginOverSecond'] >= 0.08)
]
backtest(s2b, "Longshot $10-$20 + High Conf (8%)")

# Long shots on weekdays
s2c = race_leaders[
    (race_leaders['BSP'] >= 10) & (race_leaders['BSP'] <= 20) &
    (race_leaders['DayOfWeek'] <= 4)  # Mon-Fri
]
backtest(s2c, "Longshot $10-$20 + Weekday")

# Long shots short distance
s2d = race_leaders[
    (race_leaders['BSP'] >= 10) & (race_leaders['BSP'] <= 20) &
    (race_leaders['Distance'] < 500)
]
backtest(s2d, "Longshot $10-$20 + Sprint <500m")

# Very long shots
s2e = race_leaders[
    (race_leaders['BSP'] >= 15) & (race_leaders['BSP'] <= 30) &
    (race_leaders['Tier'] >= 1)
]
backtest(s2e, "Very Long $15-$30 + Metro/Provincial")

# Long shots inside box
s2f = race_leaders[
    (race_leaders['BSP'] >= 10) & (race_leaders['BSP'] <= 20) &
    (race_leaders['Box'] <= 4)
]
backtest(s2f, "Longshot $10-$20 + Inside Box")

# Strategy 3: Combined best factors
print("\n--- STRATEGY 3: MULTI-FACTOR COMBOS ---")

# Best overall combo
s3a = race_leaders[
    (race_leaders['MarginOverSecond'] >= 0.10) &
    (race_leaders['BSP'] >= 4) & (race_leaders['BSP'] <= 8) &
    (race_leaders['Distance'] < 550) &
    (race_leaders['Tier'] >= 1)
]
backtest(s3a, "Conf 10% + $4-$8 + <550m + Metro/Prov")

# Thursday racing special
s3b = race_leaders[
    (race_leaders['DayOfWeek'] == 3) &  # Thursday
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8) &
    (race_leaders['MarginOverSecond'] >= 0.08)
]
backtest(s3b, "Thursday + $3-$8 + Conf 8%")

# Early week value
s3c = race_leaders[
    (race_leaders['DayOfWeek'] <= 2) &  # Mon-Wed
    (race_leaders['BSP'] >= 4) & (race_leaders['BSP'] <= 10) &
    (race_leaders['Distance'] < 500)
]
backtest(s3c, "Mon-Wed + $4-$10 + Sprint")

# Box 3 specialist (was +11.1% ROI)
s3d = race_leaders[
    (race_leaders['Box'] == 3) &
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)
]
backtest(s3d, "Box 3 + $3-$8")

# Middle distance specialist
s3e = race_leaders[
    (race_leaders['Distance'] >= 500) & (race_leaders['Distance'] <= 600) &
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8) &
    (race_leaders['MarginOverSecond'] >= 0.08)
]
backtest(s3e, "Mid Dist 500-600m + $3-$8 + Conf 8%")

# Strategy 4: Conservative high-volume strategies
print("\n--- STRATEGY 4: CONSERVATIVE (HIGH VOLUME) ---")

# Simple provincial
s4a = race_leaders[
    (race_leaders['Tier'] == 1) &
    (race_leaders['BSP'] >= 2.5) & (race_leaders['BSP'] <= 6)
]
backtest(s4a, "Provincial + $2.50-$6")

# Simple metro
s4b = race_leaders[
    (race_leaders['Tier'] == 2) &
    (race_leaders['BSP'] >= 2.5) & (race_leaders['BSP'] <= 6)
]
backtest(s4b, "Metro + $2.50-$6")

# Simple inside box
s4c = race_leaders[
    (race_leaders['Box'] <= 3) &
    (race_leaders['BSP'] >= 2.5) & (race_leaders['BSP'] <= 6)
]
backtest(s4c, "Box 1-3 + $2.50-$6")

print("\n" + "="*70)
print("HYBRID BACKTESTING COMPLETE")
print("="*70)
