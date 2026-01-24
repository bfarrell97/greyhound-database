"""
V18 BSP PREDICTION MODEL - Back & Lay on Price Differences
Predict final BSP, compare to Price5Min, find back/lay edge
Key insight: If we predict BSP well, we can exploit early price discrepancies
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

warnings.filterwarnings('ignore')

BETFAIR_COMMISSION = 0.10

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

print("="*70)
print("V18 BSP PREDICTION MODEL - PRICE EDGE TRADING")
print("Predict BSP, compare to Price5Min, find back/lay opportunities")
print("Train: 2020-2023 | Test: 2024 + 2025")
print("="*70)

start_time = time.time()

# Load data
conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
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
  AND ge.BSP IS NOT NULL AND ge.BSP > 1
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df = df.dropna(subset=['Position'])
df['Won'] = (df['Position'] == 1).astype(int)

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight', 'Price5Min']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
df['NormTime'] = df['FinishTime'] - df['Benchmark']
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)

# Log transform BSP for better regression (prices are log-normal)
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
df['LogPrice5Min'] = np.log(df['Price5Min'].clip(1.01, 500))

print(f"[1/6] Loaded {len(df):,} entries with BSP data")

print("[2/6] Building features with BSP history...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})

all_rows = []
processed = 0

for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 6: continue
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
            bsps = [h['bsp'] for h in recent if h['bsp'] is not None and h['bsp'] > 1]
            
            if len(times) >= 3 and len(bsps) >= 2:
                features = {
                    'RaceID': race_id, 
                    'LogBSP': r['LogBSP'],  # Target: log(BSP)
                    'BSP': r['BSP'],
                    'Price5Min': r['Price5Min'],
                    'LogPrice5Min': r['LogPrice5Min'],
                    'Won': r['Won'],
                    'Position': r['Position'],
                    'Distance': distance, 
                    'MeetingDate': race_date,
                    'Year': year
                }
                
                # ============ BSP HISTORY FEATURES (KEY) ============
                features['BSPAvg'] = np.mean(bsps)
                features['BSPAvg3'] = np.mean(bsps[-3:])
                features['BSPLag1'] = bsps[-1]
                features['BSPLag2'] = bsps[-2] if len(bsps) >= 2 else bsps[-1]
                features['BSPLag3'] = bsps[-3] if len(bsps) >= 3 else bsps[-1]
                features['BSPMin'] = min(bsps)
                features['BSPMax'] = max(bsps)
                features['BSPStd'] = np.std(bsps) if len(bsps) >= 3 else 0
                features['BSPTrend'] = bsps[-1] - bsps[0] if len(bsps) >= 2 else 0
                features['BSPTrend3'] = bsps[-1] - bsps[-3] if len(bsps) >= 3 else 0
                
                # Log BSP features
                log_bsps = [np.log(b) for b in bsps if b > 1]
                features['LogBSPAvg'] = np.mean(log_bsps)
                features['LogBSPLag1'] = log_bsps[-1]
                
                # ============ PERFORMANCE FEATURES ============
                features['TimeBest'] = min(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeStd'] = np.std(times)
                
                # Position history
                features['PosAvg'] = np.mean(positions)
                features['PosAvg3'] = np.mean(positions[-3:])
                features['PosBest'] = min(positions)
                features['PosLag1'] = positions[-1]
                features['WinRate'] = sum(1 for p in positions if p == 1) / len(positions)
                features['PlaceRate'] = sum(1 for p in positions if p <= 3) / len(positions)
                
                features['CareerStarts'] = min(len(hist), 100)
                
                # Trainer
                trainer_id = r['TrainerID']
                t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                
                # Box
                features['Box'] = int(r['Box']) if pd.notna(r['Box']) else 4
                
                # Days since
                days_since = (race_date - hist[-1]['date']).days
                features['DaysSinceRace'] = days_since
                
                # Age
                features['AgeMonths'] = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                
                all_rows.append(features)
    
    # Update history
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        won = r['Won']
        
        dog_history[dog_id].append({
            'date': race_date, 
            'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'position': r['Position'], 
            'bsp': r['BSP'] if pd.notna(r['BSP']) and r['BSP'] > 1 else None,
            'track_id': track_id, 
            'distance': distance
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
test_25_df = all_df[all_df['Year'] == 2025].copy()

# Only keep rows with Price5Min for testing (needed for edge calculation)
test_25_df = test_25_df[test_25_df['Price5Min'].notna() & (test_25_df['Price5Min'] > 1)]

exclude_cols = ['RaceID', 'LogBSP', 'BSP', 'Price5Min', 'LogPrice5Min', 'Won', 'Position', 'Distance', 'MeetingDate', 'Year']
FEATURE_COLS = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Train: {len(train_df):,} | Test 2025 (with Price5Min): {len(test_25_df):,}")
print(f"  Features: {len(FEATURE_COLS)}")

print("[3/6] Training BSP Regression Model...")
X_train = train_df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['LogBSP']  # Predict log(BSP)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Regression model for BSP prediction
model = LGBMRegressor(
    n_estimators=300,
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
test_25_df['PredLogBSP'] = model.predict(X_test_scaled)
test_25_df['PredBSP'] = np.exp(test_25_df['PredLogBSP'])

print("[4/6] Calculating BSP prediction accuracy...")

# Prediction accuracy
mae_log = np.abs(test_25_df['PredLogBSP'] - test_25_df['LogBSP']).mean()
mae_price = np.abs(test_25_df['PredBSP'] - test_25_df['BSP']).mean()
mape = (np.abs(test_25_df['PredBSP'] - test_25_df['BSP']) / test_25_df['BSP']).mean() * 100

print(f"  Log BSP MAE: {mae_log:.3f}")
print(f"  Price MAE: ${mae_price:.2f}")
print(f"  MAPE: {mape:.1f}%")

print("[5/6] Finding BACK and LAY opportunities...")

# Edge = (Price5Min / PredBSP) - 1
# If Price5Min > PredBSP → Price will shorten → BACK now
# If Price5Min < PredBSP → Price will drift → LAY now
test_25_df['PriceRatio'] = test_25_df['Price5Min'] / test_25_df['PredBSP']
test_25_df['Edge'] = test_25_df['PriceRatio'] - 1  # Positive = BACK opportunity

print("\n" + "="*70)
print("V18 RESULTS - 2025 OUT-OF-SAMPLE")
print("="*70)

def analyze_edge(df, edge_threshold, bet_type, min_bsp=1.5, max_bsp=50):
    """Analyze betting opportunities by edge"""
    if bet_type == 'BACK':
        # BACK: Price5Min > PredBSP (we expect price to shorten)
        bets = df[(df['Edge'] >= edge_threshold) & 
                  (df['Price5Min'] >= min_bsp) & (df['Price5Min'] <= max_bsp)]
    else:
        # LAY: Price5Min < PredBSP (we expect price to drift)
        bets = df[(df['Edge'] <= -edge_threshold) & 
                  (df['Price5Min'] >= min_bsp) & (df['Price5Min'] <= max_bsp)]
    
    valid = bets.dropna(subset=['BSP'])
    if len(valid) < 50:
        return None
    
    wins = valid['Won'].sum()
    sr = wins / len(valid) * 100
    
    if bet_type == 'BACK':
        # Bet at Price5Min, settle at BSP (not realistic but shows model value)
        # For realistic: we'd back at Price5Min and win/lose based on actual result
        returns = valid[valid['Won']==1]['Price5Min'].sum()
        profit = (returns * (1 - BETFAIR_COMMISSION)) - len(valid)
    else:
        # LAY at Price5Min
        losers = len(valid) - wins
        lay_profit = losers * (1 - BETFAIR_COMMISSION)
        lay_loss = (valid[valid['Won']==1]['Price5Min'] - 1).sum()
        profit = lay_profit - lay_loss
    
    roi = profit / len(valid) * 100
    
    # Also calculate average movement
    avg_move = (valid['BSP'] / valid['Price5Min']).mean() - 1
    
    return {'bets': len(valid), 'wins': wins, 'sr': sr, 'profit': profit, 'roi': roi, 'avg_move': avg_move}

print("\nBACK at Price5Min (expecting price to shorten):")
print("Edge = Price5Min is X% higher than our predicted BSP")
for edge in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = analyze_edge(test_25_df, edge, 'BACK', 2, 30)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  Edge >= {edge:.0%}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}%, AvgMove: {r['avg_move']:>+.1%} {status}")

print("\nLAY at Price5Min (expecting price to drift):")
print("Edge = Price5Min is X% lower than our predicted BSP")
for edge in [0.05, 0.10, 0.15, 0.20, 0.30]:
    r = analyze_edge(test_25_df, edge, 'LAY', 2, 30)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  Edge >= {edge:.0%}: {r['bets']:>5} bets, WinRate: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}%, AvgMove: {r['avg_move']:>+.1%} {status}")

print("\nBACK by Price Range (15% edge):")
for low, high in [(2, 5), (5, 10), (10, 20), (20, 50)]:
    r = analyze_edge(test_25_df, 0.15, 'BACK', low, high)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  ${low}-${high}: {r['bets']:>5} bets, SR: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

print("\nLAY by Price Range (15% edge):")
for low, high in [(2, 5), (5, 10), (10, 20), (20, 50)]:
    r = analyze_edge(test_25_df, 0.15, 'LAY', low, high)
    if r:
        status = "✓" if r['roi'] >= 5 else ""
        print(f"  ${low}-${high}: {r['bets']:>5} bets, WinRate: {r['sr']:>5.1f}%, ROI: {r['roi']:>+6.1f}% {status}")

# Feature importance
print("\n[6/6] Top BSP Prediction Features:")
importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

print(f"\nTotal Time: {(time.time() - start_time)/60:.1f} minutes")

# Save model
print("\nSaving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'features': FEATURE_COLS
}
with open('models/pace_v18_bsp.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Saved to models/pace_v18_bsp.pkl")
print("="*70)
