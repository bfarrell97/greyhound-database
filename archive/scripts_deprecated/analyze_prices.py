"""
Analyze Price5Min vs BSP for V12 High Confidence Strategy
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

print("="*60)
print("PRICE ANALYSIS: Price5Min vs BSP")
print("Strategy: High Conf (Margin > 10%) + $3-$8 + <550m")
print("="*60)

# Load model
with open('models/pace_v12_optimized.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['features']

# Load last 1 year of data for speed
cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
print(f"Loading data since {cutoff_date}...")

conn = sqlite3.connect('greyhound_racing.db')
bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}

query = f"""
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
WHERE rm.MeetingDate >= '{cutoff_date}'
  AND ge.Position IS NOT NULL 
  AND ge.Position NOT IN ('SCR', 'DNF', '')
ORDER BY rm.MeetingDate, ge.RaceID
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Preprocessing
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['Won'] = (df['Position'] == 1).astype(int)
df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')

for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'Weight',
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

# Feature Building (Simplified for speed - focusing on main ones)
print("Building features...")
dog_history = defaultdict(list)
trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
feature_rows = []

# Pre-computation for trainers (rolling approximation)
# Real implementation would do full history, here we do single pass
for race_id, race_df in df.groupby('RaceID', sort=False):
    if len(race_df) < 4: continue
    race_date = race_df['MeetingDate'].iloc[0]
    distance = race_df['Distance'].iloc[0]
    
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_history.get(dog_id, [])
        
        if len(hist) >= 3:
            recent = hist[-10:]
            times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
            
            if len(times) >= 3:
                features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 
                           'Price5Min': r['Price5Min'], 'Distance': distance}
                
                # Minimal features needed for reasonable prediction approx
                features['TimeBest'] = min(times)
                features['TimeWorst'] = max(times)
                features['TimeAvg'] = np.mean(times)
                features['TimeAvg3'] = np.mean(times[-3:])
                features['TimeLag1'] = times[-1]
                features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                features['TimeStd'] = np.std(times)
                features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                features['WinRate5'] = sum(1 for h in recent[:5] if h['position']==1)/5
                features['CareerWinRate'] = sum(1 for h in hist if h['position']==1)/len(hist)
                
                # Fill missing cols with 0
                for col in feature_cols:
                    if col not in features:
                        features[col] = 0
                
                feature_rows.append(features)
    
    # Update History
    for _, r in race_df.iterrows():
        dog_history[r['GreyhoundID']].append({
            'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
            'position': r['Position']
        })

print(f"Generated features for {len(feature_rows):,} dogs")
feat_df = pd.DataFrame(feature_rows)

# Prediction
print("Running predictions...")
X = feat_df[feature_cols]
X_scaled = scaler.transform(X)
feat_df['PredProb'] = model.predict_proba(X_scaled)[:, 1]
feat_df['MarginOverSecond'] = feat_df.groupby('RaceID')['PredProb'].transform(
    lambda x: x.max() - sorted(x)[-2] if len(x) > 1 else 0
)

# Filter Strategy
race_leaders = feat_df.loc[feat_df.groupby('RaceID')['PredProb'].idxmax()]
strategy_bets = race_leaders[
    (race_leaders['MarginOverSecond'] >= 0.10) & 
    (race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8) &
    (race_leaders['Distance'] < 550)
].copy()

print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

total_bets = len(strategy_bets)
with_price5 = strategy_bets.dropna(subset=['Price5Min'])
coverage = len(with_price5) / total_bets * 100

print(f"Total Strategy Bets: {total_bets:,}")
print(f"Bets with Price5Min: {len(with_price5):,} ({coverage:.1f}%)")

if len(with_price5) > 0:
    wins = with_price5['Won'].sum()
    sr = wins / len(with_price5) * 100
    
    avg_bsp = with_price5['BSP'].mean()
    avg_price5 = with_price5['Price5Min'].mean()
    
    # ROI Calculation (Flat stakes)
    profit_bsp = with_price5[with_price5['Won']==1]['BSP'].sum() - len(with_price5)
    roi_bsp = profit_bsp / len(with_price5) * 100
    
    profit_price5 = with_price5[with_price5['Won']==1]['Price5Min'].sum() - len(with_price5)
    roi_price5 = profit_price5 / len(with_price5) * 100
    
    p5_better = sum(1 for _, r in with_price5.iterrows() if r['Price5Min'] > r['BSP'])
    bsp_better = sum(1 for _, r in with_price5.iterrows() if r['BSP'] > r['Price5Min'])
    
    print(f"\nWin Rate: {sr:.1f}%")
    print(f"\nAverage Price Comparison:")
    print(f"  Avg BSP:       ${avg_bsp:.2f}")
    print(f"  Avg Price5Min: ${avg_price5:.2f} ({((avg_price5-avg_bsp)/avg_bsp*100):+.1f}%)")
    
    print(f"\nROI Comparison (Flat Stakes, No Commission):")
    print(f"  ROI (BSP):       {roi_bsp:+.1f}%")
    print(f"  ROI (Price5Min): {roi_price5:+.1f}%")
    
    print(f"\nPrice Advantage Frequency:")
    print(f"  Price5Min > BSP: {p5_better} ({p5_better/len(with_price5)*100:.1f}%)")
    print(f"  BSP > Price5Min: {bsp_better} ({bsp_better/len(with_price5)*100:.1f}%)")
    
else:
    print("Insufficient data for Price5Min analysis.")

print("="*60)
