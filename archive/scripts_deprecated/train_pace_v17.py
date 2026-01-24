"""
V17 Position Prediction Model - Back & Lay Edge Detection
Instead of predicting winner, we predict finish position (1-8)
Then convert to win probability and compare to market for back/lay edges
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
import time
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

print("="*70)
print("V17 POSITION PREDICTION MODEL - BACK & LAY")
print("Predict finish position (1-8), find back/lay edge")
print("Train: 2020-2023 | Test: 2024 + 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
       ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
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

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)

print(f"[1/6] Loaded {len(df):,} entries")

print("[2/6] Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})

all_rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 6: continue  # Need 6+ dogs for reliable position prediction
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    track_id = race_df['TrackID'].iloc[0]
    year = race_date.year
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_history.get(dog_id, [])
        
        if len(hist) >= 3:
            recent = hist[-10:]
            times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
            positions = [h['position'] for h in recent if h['position'] is not None]
            splits = [h['split'] for h in recent if h['split'] is not None]
            
            if len(times) >= 3 and len(positions) >= 3:
                features = {
                    'RaceID': race_id, 
                    'Position': r['Position'],  # Target: actual finish position
                    'Won': r['Won'],
                    'BSP': r['BSP'],
                    'Distance': distance, 
                    'MeetingDate': race_date,
                    'Year': year
                }
                
                # Time Features
                features['TimeBest'] = min(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeStd'] = np.std(times)
                
                features['SplitBest'] = min(splits) if splits else 0
                features['SplitAvg'] = np.mean(splits) if splits else 0
                
                # Position history - KEY for position prediction
                features['PosAvg'] = np.mean(positions)
                features['PosAvg3'] = np.mean(positions[-3:])
                features['PosBest'] = min(positions)
                features['PosLag1'] = positions[-1]
                features['PosStd'] = np.std(positions)
                features['WinRate'] = sum(1 for p in positions if p == 1) / len(positions)
                features['PlaceRate'] = sum(1 for p in positions if p <= 3) / len(positions)
                features['Top4Rate'] = sum(1 for p in positions if p <= 4) / len(positions)
                
                features['CareerStarts'] = min(len(hist), 100)
                
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                box = int(r['Box']) if pd.notna(r['Box']) else 4
                features['Box'] = box
                
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                
                all_rows.append(features)
    
    # Update history
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        won = r['Won']
        
        dog_history[dog_id].append({
            'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'position': r['Position'], 'track_id': track_id, 'distance': distance,
            'split': r['Split'] if pd.notna(r['Split']) else None
        })
        
        if pd.notna(r['TrainerID']):
            tid = r['TrainerID']
            trainer_all[tid]['runs'] += 1
            if won: trainer_all[tid]['wins'] += 1
    
    processed += 1
    if processed % 50000 == 0: print(f"  {processed:,} races...")

all_df = pd.DataFrame(all_rows)

# Split by year
train_df = all_df[all_df['Year'] <= 2023].copy()
test_24_df = all_df[all_df['Year'] == 2024].copy()
test_25_df = all_df[all_df['Year'] == 2025].copy()

exclude_cols = ['RaceID', 'Position', 'Won', 'BSP', 'Distance', 'MeetingDate', 'Year']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} | Test 2024: {len(test_24_df):,} | Test 2025: {len(test_25_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

print("[3/6] Training Position Regression Model...")
X_train = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['Position']  # Continuous target: 1-8

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Regression model for position prediction
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=40,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train_scaled, y_train)

# Predict on 2025 test set
X_test = test_25_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X_test_scaled = scaler.transform(X_test)
test_25_df['PredPos'] = model.predict(X_test_scaled)

print("[4/6] Converting position predictions to win probabilities...")

def position_to_prob(pred_pos, total_dogs=8):
    """
    Convert predicted position to win probability
    Lower predicted position = higher win probability
    Uses inverse relationship: prob ∝ 1/position
    """
    # Clip to valid range
    pos = max(1, min(pred_pos, total_dogs))
    # Inverse position gives rough probability
    # Position 1 → high prob, Position 8 → low prob
    raw_prob = 1 / pos
    return raw_prob

# Calculate predicted win probability for each dog
test_25_df['RawProb'] = test_25_df['PredPos'].apply(position_to_prob)

# Normalize probabilities within each race to sum to 1
test_25_df['PredProb'] = test_25_df.groupby('RaceID')['RawProb'].transform(lambda x: x / x.sum())

# Market implied probability from BSP
test_25_df['MarketProb'] = 1 / test_25_df['BSP'].clip(1.01, 100)

# Edge = Our probability - Market probability
test_25_df['Edge'] = test_25_df['PredProb'] - test_25_df['MarketProb']

print("[5/6] Finding BACK and LAY opportunities...")

print("\n" + "="*70)
print("V17 RESULTS - 2025 OUT-OF-SAMPLE")
print("="*70)

def analyze_edge(df, edge_threshold, bet_type, min_bsp=1.5, max_bsp=50):
    """Analyze betting opportunities by edge"""
    if bet_type == 'BACK':
        # BACK: We think dog is undervalued (our prob > market prob)
        bets = df[(df['Edge'] >= edge_threshold) & 
                  (df['BSP'] >= min_bsp) & (df['BSP'] <= max_bsp)]
    else:
        # LAY: We think dog is overvalued (our prob < market prob)
        bets = df[(df['Edge'] <= -edge_threshold) & 
                  (df['BSP'] >= min_bsp) & (df['BSP'] <= max_bsp)]
    
    valid = bets.dropna(subset=['BSP'])
    if len(valid) < 50:
        return None
    
    wins = valid['Won'].sum()
    sr = wins / len(valid) * 100
    
    if bet_type == 'BACK':
        # Profit from backing winners at BSP
        returns = valid[valid['Won']==1]['BSP'].sum()
        profit = (returns * (1 - BETFAIR_COMMISSION)) - len(valid)
    else:
        # Profit from laying losers (liability on winners)
        losers = len(valid) - wins
        # Lay profit = stakes from losers * commission adjustment
        # Lay loss = (BSP - 1) for each winner
        lay_profit = losers * (1 - BETFAIR_COMMISSION)
        lay_loss = (valid[valid['Won']==1]['BSP'] - 1).sum()
        profit = lay_profit - lay_loss
    
    roi = profit / len(valid) * 100
    return {'bets': len(valid), 'wins': wins, 'sr': sr, 'profit': profit, 'roi': roi}

print("\nBACK Opportunities (our prob > market + edge):")
for edge in [0.02, 0.05, 0.08, 0.10, 0.15]:
    r = analyze_edge(test_25_df, edge, 'BACK')
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  Edge >= {edge:.0%}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

print("\nLAY Opportunities (our prob < market - edge):")
for edge in [0.02, 0.05, 0.08, 0.10, 0.15]:
    r = analyze_edge(test_25_df, edge, 'LAY')
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  Edge >= {edge:.0%}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}% L, ROI: {r['roi']:>+6.1f}% {status}")

print("\nBACK by Price Range (10% edge):")
for low, high in [(1.5, 3), (3, 6), (6, 15), (15, 50)]:
    r = analyze_edge(test_25_df, 0.10, 'BACK', low, high)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  ${low:.1f}-${high}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

print("\nLAY by Price Range (10% edge):")
for low, high in [(1.5, 3), (3, 6), (6, 15), (15, 50)]:
    r = analyze_edge(test_25_df, 0.10, 'LAY', low, high)
    if r:
        # For LAY, low SR is GOOD (we want losers)
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  ${low:.1f}-${high}: {r['bets']:>5} bets, Win: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

# Position prediction accuracy
print("\n[6/6] Position Prediction Accuracy:")
mae = np.abs(test_25_df['PredPos'] - test_25_df['Position']).mean()
print(f"  Mean Absolute Error: {mae:.2f} positions")

# Check if predicted favorites actually win more
test_25_df['IsPredFav'] = test_25_df.groupby('RaceID')['PredPos'].transform('min') == test_25_df['PredPos']
pred_favs = test_25_df[test_25_df['IsPredFav']]
fav_sr = pred_favs['Won'].mean() * 100
print(f"  Predicted Favorite Win Rate: {fav_sr:.1f}%")

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Save model
print("\nSaving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'features': FEATURE_COLS
}
with open('models/pace_v17_position.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved to models/pace_v17_position.pkl")
print("="*70)
