"""
Strategy Exploration with Proper Holdout
Train: 2020-2024 | Test: 2025
Goal: Find a strategy that achieves positive ROI with 10% commission
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

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

print("="*70)
print("STRATEGY EXPLORATION (Train 2020-2024, Test 2025)")
print("Goal: Find profitable strategy with 10% commission")
print("="*70)

# Load saved predictions from holdout test (if available) or rebuild
# For speed, let's load from the validate_holdout run if we saved it
# Otherwise we need to rebuild - let's check if we can load

try:
    # Try to load cached predictions
    test_feat_df = pd.read_pickle('temp_2025_predictions.pkl')
    print("[CACHE] Loaded cached 2025 predictions")
except:
    print("[INFO] Need to rebuild predictions - running full pipeline...")
    
    with open('models/pace_v12_optimized.pkl', 'rb') as f:
        model_data = pickle.load(f)
        feature_cols = model_data['features']
        best_params = model_data['params']
    
    conn = sqlite3.connect('greyhound_racing.db')
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
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
    df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek
    df['Tier'] = df['TrackName'].apply(get_tier)
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Price5Min', 'Weight',
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
    
    print(f"Loaded {len(df):,} entries")
    
    # Feature building (simplified for speed - key features only)
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    
    train_rows = []
    test_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4: continue
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        track_name = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        day_of_week = race_df['DayOfWeek'].iloc[0]
        is_test = race_date.year == 2025
        
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
                        'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Price5Min': r['Price5Min'],
                        'Distance': distance, 'MeetingDate': race_date, 'TrackName': track_name,
                        'Tier': tier, 'DayOfWeek': day_of_week, 'Box': int(r['Box']) if pd.notna(r['Box']) else 4
                    }
                    
                    # Core features
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
                    
                    form_trend = 0
                    if len(positions) >= 5:
                        first_half = np.mean(positions[:len(positions)//2])
                        second_half = np.mean(positions[len(positions)//2:])
                        form_trend = first_half - second_half
                    features['FormTrend'] = form_trend
                    
                    trainer_id = r['TrainerID']
                    t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                    features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                    
                    dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                    features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                    
                    track_runs = [h for h in hist if h['track_id'] == track_id]
                    features['TrackWinRate'] = safe_div(sum(1 for t in track_runs if t['position'] == 1), len(track_runs), features['CareerWinRate']) if len(track_runs) >= 3 else features['CareerWinRate']
                    
                    age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                    features['AgeMonths'] = age_months
                    
                    days_since = (race_date - hist[-1]['date']).days
                    features['DaysSinceRace'] = days_since
                    features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                    
                    # Fill missing
                    for col in feature_cols:
                        if col not in features:
                            features[col] = 0
                    
                    if is_test:
                        test_rows.append(features)
                    else:
                        train_rows.append(features)
        
        # Update history
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            box = int(r['Box']) if pd.notna(r['Box']) else 4
            won = r['Won']
            
            dog_history[dog_id].append({
                'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
                'position': r['Position'], 'track_id': track_id, 'distance': distance,
                'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
                'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
                'weight': r['Weight'], 'box': box, 'split': r['Split'] if pd.notna(r['Split']) else None
            })
            
            if pd.notna(r['TrainerID']):
                tid = r['TrainerID']
                trainer_all[tid]['runs'] += 1
                if won: trainer_all[tid]['wins'] += 1
        
        processed += 1
        if processed % 50000 == 0: print(f"  {processed:,} races...")
    
    train_feat_df = pd.DataFrame(train_rows)
    test_feat_df = pd.DataFrame(test_rows)
    
    print(f"  Train: {len(train_feat_df):,}, Test: {len(test_feat_df):,}")
    
    # Train model
    print("Training model...")
    X_train = train_feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_feat_df['Won']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test
    print("Predicting on 2025...")
    X_test = test_feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_scaled = scaler.transform(X_test)
    
    test_feat_df['PredProb'] = model.predict_proba(X_test_scaled)[:, 1]
    test_feat_df['MarginOverSecond'] = test_feat_df.groupby('RaceID')['PredProb'].transform(
        lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
    )
    
    # Save for reuse
    test_feat_df.to_pickle('temp_2025_predictions.pkl')
    print("Saved predictions to cache")

# Get race leaders (highest prob per race)
race_leaders = test_feat_df.loc[test_feat_df.groupby('RaceID')['PredProb'].idxmax()].copy()
race_leaders['BSP'] = pd.to_numeric(race_leaders['BSP'], errors='coerce')
print(f"\nTotal 2025 Race Leaders: {len(race_leaders):,}")

def test_strategy(df, name):
    """Test a strategy and return results"""
    if len(df) < 30:
        return None
    
    valid = df.dropna(subset=['BSP'])
    wins = valid['Won'].sum()
    sr = wins / len(valid) * 100
    
    returns = valid[valid['Won']==1]['BSP'].sum()
    profit = returns - len(valid)
    profit_comm = (returns * (1 - BETFAIR_COMMISSION)) - len(valid)
    roi = profit / len(valid) * 100
    roi_comm = profit_comm / len(valid) * 100
    
    return {
        'Name': name,
        'Bets': len(valid),
        'Wins': wins,
        'SR': sr,
        'ROI': roi,
        'ROI_Comm': roi_comm,
        'Profit_Comm': profit_comm
    }

print("\n" + "="*70)
print("STRATEGY EXPLORATION")
print("="*70)

results = []

# Base strategies by confidence
for conf in [0.05, 0.08, 0.10, 0.12, 0.15]:
    strat = race_leaders[race_leaders['MarginOverSecond'] >= conf]
    r = test_strategy(strat, f"Confidence >= {conf:.0%}")
    if r: results.append(r)

# Price ranges
for low, high in [(2, 5), (3, 6), (3, 8), (4, 10), (5, 15), (10, 30)]:
    strat = race_leaders[(race_leaders['BSP'] >= low) & (race_leaders['BSP'] <= high)]
    r = test_strategy(strat, f"Price ${low}-${high}")
    if r: results.append(r)

# Distance ranges
for low, high in [(0, 450), (450, 520), (500, 550), (550, 700), (700, 900)]:
    strat = race_leaders[(race_leaders['Distance'] >= low) & (race_leaders['Distance'] < high)]
    r = test_strategy(strat, f"Distance {low}-{high}m")
    if r: results.append(r)

# Box
for box in [1, 2, 3, 4, 5, 6, 7, 8]:
    strat = race_leaders[race_leaders['Box'] == box]
    r = test_strategy(strat, f"Box {box}")
    if r: results.append(r)

# Day of week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for day in range(7):
    strat = race_leaders[race_leaders['DayOfWeek'] == day]
    r = test_strategy(strat, f"{days[day]}")
    if r: results.append(r)

# Tier
for tier, name in [(0, 'Country'), (1, 'Provincial'), (2, 'Metro')]:
    strat = race_leaders[race_leaders['Tier'] == tier]
    r = test_strategy(strat, f"Tier: {name}")
    if r: results.append(r)

# Combinations with confidence
for conf in [0.08, 0.10]:
    # Conf + Price
    for low, high in [(3, 6), (3, 8), (4, 10)]:
        strat = race_leaders[
            (race_leaders['MarginOverSecond'] >= conf) & 
            (race_leaders['BSP'] >= low) & (race_leaders['BSP'] <= high)
        ]
        r = test_strategy(strat, f"Conf {conf:.0%} + ${low}-${high}")
        if r: results.append(r)
    
    # Conf + Distance
    for low, high in [(0, 500), (450, 550), (500, 600)]:
        strat = race_leaders[
            (race_leaders['MarginOverSecond'] >= conf) & 
            (race_leaders['Distance'] >= low) & (race_leaders['Distance'] < high)
        ]
        r = test_strategy(strat, f"Conf {conf:.0%} + {low}-{high}m")
        if r: results.append(r)

# Triple combinations
for conf in [0.08, 0.10]:
    for low_p, high_p in [(3, 6), (3, 8)]:
        for low_d, high_d in [(0, 500), (450, 550)]:
            strat = race_leaders[
                (race_leaders['MarginOverSecond'] >= conf) & 
                (race_leaders['BSP'] >= low_p) & (race_leaders['BSP'] <= high_p) &
                (race_leaders['Distance'] >= low_d) & (race_leaders['Distance'] < high_d)
            ]
            r = test_strategy(strat, f"Conf {conf:.0%} + ${low_p}-${high_p} + {low_d}-{high_d}m")
            if r: results.append(r)

# Metro/Provincial only combinations
for tier in [1, 2]:
    tier_name = 'Provincial' if tier == 1 else 'Metro'
    for conf in [0.08, 0.10]:
        for low, high in [(3, 8), (4, 10)]:
            strat = race_leaders[
                (race_leaders['Tier'] == tier) &
                (race_leaders['MarginOverSecond'] >= conf) & 
                (race_leaders['BSP'] >= low) & (race_leaders['BSP'] <= high)
            ]
            r = test_strategy(strat, f"{tier_name} + Conf {conf:.0%} + ${low}-${high}")
            if r: results.append(r)

# Weekday combinations
for conf in [0.08, 0.10]:
    for low, high in [(3, 8), (4, 10)]:
        strat = race_leaders[
            (race_leaders['DayOfWeek'] <= 4) &  # Mon-Fri
            (race_leaders['MarginOverSecond'] >= conf) & 
            (race_leaders['BSP'] >= low) & (race_leaders['BSP'] <= high)
        ]
        r = test_strategy(strat, f"Weekday + Conf {conf:.0%} + ${low}-${high}")
        if r: results.append(r)

# Sort by ROI with commission
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROI_Comm', ascending=False)

print("\n" + "="*70)
print("TOP 20 STRATEGIES (By ROI with 10% Commission)")
print("="*70)
print(f"{'Strategy':<45} {'Bets':>6} {'SR%':>6} {'ROI':>8} {'ROI-C':>8} {'Profit':>10}")
print("-"*85)

for _, row in results_df.head(20).iterrows():
    print(f"{row['Name']:<45} {row['Bets']:>6} {row['SR']:>5.1f}% {row['ROI']:>+7.1f}% {row['ROI_Comm']:>+7.1f}% ${row['Profit_Comm']:>+8.0f}")

print("\n" + "="*70)
print("PROFITABLE STRATEGIES (ROI with Commission > 0)")
print("="*70)

profitable = results_df[results_df['ROI_Comm'] > 0]
if len(profitable) > 0:
    for _, row in profitable.iterrows():
        print(f"{row['Name']:<45} {row['Bets']:>6} {row['SR']:>5.1f}% {row['ROI_Comm']:>+7.1f}% ${row['Profit_Comm']:>+8.0f}")
else:
    print("No strategies found with positive ROI after commission")

print("="*70)
