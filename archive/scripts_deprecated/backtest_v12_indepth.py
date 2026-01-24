"""
V12 In-Depth Backtesting - Multiple Scenarios & Combinations
=============================================================
Tests various betting strategies, filters, and conditions.
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
    """Calculate backtest metrics"""
    if len(df) < min_bets: 
        return None
    wins = df['Won'].sum()
    sr = wins / len(df) * 100
    valid = df.dropna(subset=['BSP'])
    if len(valid) == 0:
        return None
    returns = valid[valid['Won'] == 1]['BSP'].sum()
    profit = returns - len(valid)
    roi = profit / len(valid) * 100
    avg_bsp = valid['BSP'].mean()
    return {
        'Label': label,
        'Bets': len(df),
        'Wins': wins,
        'StrikeRate': sr,
        'ROI': roi,
        'Profit': profit,
        'AvgBSP': avg_bsp
    }

def print_result(result):
    if result:
        print(f"{result['Label']}: {result['Bets']:,} bets, {result['Wins']:,} wins ({result['StrikeRate']:.1f}%), ROI: {result['ROI']:+.1f}%, Profit: {result['Profit']:+.1f}")
    
def print_results_table(results):
    """Print results in a nice table format"""
    if not results:
        return
    df = pd.DataFrame([r for r in results if r is not None])
    if len(df) == 0:
        return
    df = df.sort_values('ROI', ascending=False)
    print(df.to_string(index=False))

print("="*70)
print("V12 IN-DEPTH BACKTESTING")
print("="*70)

# Load model
print("\n[1/5] Loading V12 model...")
with open('models/pace_v12_optimized.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['features']
print(f"  CV AUC: {model_data['cv_auc']:.4f}, Features: {len(feature_cols)}")

# Load test data
print("\n[2/5] Loading test data...")
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
df['Month'] = df['MeetingDate'].dt.to_period('M')

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

# Distance categories
df['DistCat'] = pd.cut(df['Distance'], bins=[0, 400, 500, 600, 1000], labels=['Sprint(<400m)', 'Short(400-500m)', 'Middle(500-600m)', 'Stay(600m+)'])

print(f"  Loaded {len(df):,} entries")

# Build features (same as quick backtest but captures more metadata)
print("\n[3/5] Building features...")
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
    if len(race_df) < 4:
        continue
    
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    track_id = race_df['TrackID'].iloc[0]
    track_name = race_df['TrackName'].iloc[0]
    tier = race_df['Tier'].iloc[0]
    day_of_week = race_df['DayOfWeek'].iloc[0]
    month = race_df['Month'].iloc[0]
    dist_cat = race_df['DistCat'].iloc[0]
    
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
                features = {
                    'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 
                    'Tier': tier, 'TrackName': track_name, 'Distance': distance,
                    'DayOfWeek': day_of_week, 'Month': str(month), 'DistCat': dist_cat,
                    'Box': int(r['Box'])
                }
                
                # Core features (same as before - abbreviated for space)
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
                
                if len(positions) >= 5:
                    first_half = np.mean(positions[:len(positions)//2])
                    second_half = np.mean(positions[len(positions)//2:])
                    form_trend = first_half - second_half
                else:
                    form_trend = 0
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
                
                # Derived features
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

print(f"  Total: {processed:,} races, {len(feature_rows):,} features")

# Predict
print("\n[4/5] Running V12 predictions...")
feat_df = pd.DataFrame(feature_rows)
feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')

model_features = [c for c in feature_cols if c in feat_df.columns]
for col in feature_cols:
    if col not in feat_df.columns:
        feat_df[col] = 0

X = feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_scaled = scaler.transform(X)
feat_df['PredProb'] = model.predict_proba(X_scaled)[:, 1]

# Calculate confidence percentile within each race
feat_df['RaceMaxProb'] = feat_df.groupby('RaceID')['PredProb'].transform('max')
feat_df['RaceMinProb'] = feat_df.groupby('RaceID')['PredProb'].transform('min')
feat_df['ConfidenceRatio'] = feat_df['PredProb'] / feat_df['RaceMaxProb']
feat_df['MarginOverSecond'] = feat_df.groupby('RaceID')['PredProb'].transform(lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0)

# Pick top dog per race
race_leaders = feat_df.loc[feat_df.groupby('RaceID')['PredProb'].idxmax()].copy()

print(f"  Generated {len(race_leaders):,} race predictions")

# ==============================================================
# SECTION 5: IN-DEPTH BACKTESTING
# ==============================================================
print("\n" + "="*70)
print("[5/5] IN-DEPTH BACKTESTING RESULTS")
print("="*70)

# ----- 5.1: BY PRICE RANGE -----
print("\n--- 5.1: BY PRICE RANGE ---")
price_ranges = [
    (1.01, 2.00, "$1.01-$2.00 (Short)"),
    (2.00, 3.00, "$2-$3 (Favorites)"),
    (3.00, 4.00, "$3-$4"),
    (4.00, 5.00, "$4-$5"),
    (5.00, 6.00, "$5-$6"),
    (6.00, 8.00, "$6-$8"),
    (8.00, 10.00, "$8-$10"),
    (10.00, 15.00, "$10-$15 (Value)"),
    (15.00, 50.00, "$15-$50 (Long shot)")
]

for low, high, label in price_ranges:
    subset = race_leaders[(race_leaders['BSP'] >= low) & (race_leaders['BSP'] < high)]
    print_result(backtest(subset, label))

# ----- 5.2: BY TRACK TIER -----
print("\n--- 5.2: BY TRACK TIER (All Prices) ---")
tiers = [(2, "Metro"), (1, "Provincial"), (0, "Country")]
for tier_val, label in tiers:
    subset = race_leaders[race_leaders['Tier'] == tier_val]
    print_result(backtest(subset, label))

# ----- 5.3: BY DISTANCE CATEGORY -----
print("\n--- 5.3: BY DISTANCE CATEGORY ---")
for dist_cat in race_leaders['DistCat'].dropna().unique():
    subset = race_leaders[race_leaders['DistCat'] == dist_cat]
    print_result(backtest(subset, str(dist_cat)))

# ----- 5.4: BY DAY OF WEEK -----
print("\n--- 5.4: BY DAY OF WEEK ---")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(days):
    subset = race_leaders[race_leaders['DayOfWeek'] == i]
    print_result(backtest(subset, day))

# ----- 5.5: BY BOX NUMBER -----
print("\n--- 5.5: BY BOX NUMBER (When Selected) ---")
for box in range(1, 9):
    subset = race_leaders[race_leaders['Box'] == box]
    print_result(backtest(subset, f"Box {box}"))

# ----- 5.6: BY MODEL CONFIDENCE -----
print("\n--- 5.6: BY MODEL CONFIDENCE (Margin Over 2nd) ---")
margins = [
    (0.00, 0.05, "Low (<5% margin)"),
    (0.05, 0.10, "Medium (5-10% margin)"),
    (0.10, 0.15, "High (10-15% margin)"),
    (0.15, 0.20, "Very High (15-20% margin)"),
    (0.20, 1.00, "Extreme (>20% margin)")
]
for low, high, label in margins:
    subset = race_leaders[(race_leaders['MarginOverSecond'] >= low) & (race_leaders['MarginOverSecond'] < high)]
    result = backtest(subset, label)
    print_result(result)

# ----- 5.7: COMBINED FILTERS (Best Scenarios) -----
print("\n--- 5.7: COMBINED FILTERS (Optimal Scenarios) ---")

scenarios = [
    ("Metro + $3-$8", (race_leaders['Tier'] == 2) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Metro + $4-$6", (race_leaders['Tier'] == 2) & (race_leaders['BSP'] >= 4) & (race_leaders['BSP'] <= 6)),
    ("Provincial + $3-$8", (race_leaders['Tier'] == 1) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Provincial + $4-$6", (race_leaders['Tier'] == 1) & (race_leaders['BSP'] >= 4) & (race_leaders['BSP'] <= 6)),
    ("High Confidence + $3-$8", (race_leaders['MarginOverSecond'] >= 0.10) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("High Confidence + Metro", (race_leaders['MarginOverSecond'] >= 0.10) & (race_leaders['Tier'] == 2)),
    ("High Confidence + Provincial", (race_leaders['MarginOverSecond'] >= 0.10) & (race_leaders['Tier'] == 1)),
    ("Weekday + $3-$8", (race_leaders['DayOfWeek'] <= 4) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Weekend Metro", (race_leaders['DayOfWeek'] >= 5) & (race_leaders['Tier'] == 2)),
    ("Sprint Distance + $3-$8", (race_leaders['Distance'] < 450) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Middle Distance + $3-$8", ((race_leaders['Distance'] >= 500) & (race_leaders['Distance'] <= 600)) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Inside Box (1-4) + $3-$8", (race_leaders['Box'] <= 4) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
    ("Outside Box (5-8) + $3-$8", (race_leaders['Box'] >= 5) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)),
]

results = []
for label, condition in scenarios:
    subset = race_leaders[condition]
    result = backtest(subset, label)
    if result:
        results.append(result)

print_results_table(results)

# ----- 5.8: LAY BETTING SCENARIOS -----
print("\n--- 5.8: LAY BETTING (Backing Losers to Lose) ---")

# For lay betting we want dogs with LOW win probability that are short priced
feat_df['RaceRank'] = feat_df.groupby('RaceID')['PredProb'].rank(ascending=False)

# Lay the favorite when model says it shouldn't be
bad_favorites = feat_df[(feat_df['RaceRank'] > 1) & (feat_df['BSP'] < 3)]  # Not top pick but short priced
lay_scenarios = [
    ("Lay $1.50-$2.00 (Model says no)", bad_favorites[(bad_favorites['BSP'] >= 1.50) & (bad_favorites['BSP'] < 2.00)]),
    ("Lay $2.00-$2.50 (Model says no)", bad_favorites[(bad_favorites['BSP'] >= 2.00) & (bad_favorites['BSP'] < 2.50)]),
    ("Lay $2.50-$3.00 (Model says no)", bad_favorites[(bad_favorites['BSP'] >= 2.50) & (bad_favorites['BSP'] < 3.00)]),
]

for label, subset in lay_scenarios:
    if len(subset) < 50:
        continue
    # For lays: Win = dog lost, Profit = BSP-1 when lose, Loss = 1 when win
    losses = subset[subset['Won'] == 1]  # We lose when dog wins
    wins = subset[subset['Won'] == 0]  # We win when dog loses
    
    if len(losses) > 0:
        lay_liability = losses['BSP'].sum() - len(losses)  # How much we lost
    else:
        lay_liability = 0
    lay_profit = len(wins) - lay_liability
    lay_roi = lay_profit / len(subset) * 100 if len(subset) > 0 else 0
    strike_rate = len(wins) / len(subset) * 100
    print(f"{label}: {len(subset):,} lays, {len(wins):,} wins ({strike_rate:.1f}%), ROI: {lay_roi:+.1f}%, Profit: {lay_profit:+.1f}")

# ----- 5.9: MONTHLY BREAKDOWN -----
print("\n--- 5.9: MONTHLY BREAKDOWN ($3-$8 Range) ---")
monthly = race_leaders[(race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)]
months = sorted(monthly['Month'].unique())
monthly_results = []
for month in months:
    subset = monthly[monthly['Month'] == month]
    result = backtest(subset, str(month), min_bets=10)
    if result:
        monthly_results.append(result)

for result in monthly_results:
    print_result(result)

# ----- 5.10: TOP TRACKS -----
print("\n--- 5.10: TOP PERFORMING TRACKS ($3-$8 Range) ---")
track_results = []
for track in race_leaders['TrackName'].unique():
    subset = race_leaders[(race_leaders['TrackName'] == track) & (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)]
    result = backtest(subset, track)
    if result and result['Bets'] >= 100:
        track_results.append(result)

track_df = pd.DataFrame(track_results)
if len(track_df) > 0:
    track_df = track_df.sort_values('ROI', ascending=False)
    print(track_df.head(15).to_string(index=False))

print("\n" + "="*70)
print("BACKTESTING COMPLETE")
print("="*70)
