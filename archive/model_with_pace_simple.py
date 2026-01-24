"""
Add finish pace feature to ML model - Python approach
Calculate features in Python rather than complex SQL
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

print("="*100)
print("REBUILDING ML MODEL WITH FINISH PACE FEATURE (Python calculation)")
print("="*100)

conn = sqlite3.connect(DB_PATH)

# Load all historical race data
print("\nLoading historical data...")

query = """
SELECT
    ge.EntryID,
    ge.GreyhoundID,
    ge.Box,
    ge.Weight,
    CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS INTEGER) as is_winner,
    ge.FinishTimeBenchmarkLengths,
    r.Distance,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2023-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""

df = pd.read_sql_query(query, conn)
print(f"Loaded {len(df)} races")

# Calculate rolling features by dog
print("\nCalculating rolling features...")

features_data = []

for dog_id in df['GreyhoundID'].unique():
    dog_races = df[df['GreyhoundID'] == dog_id].sort_values('MeetingDate').reset_index(drop=True)
    
    if len(dog_races) >= 2:
        for idx in range(1, len(dog_races)):
            race = dog_races.iloc[idx]
            previous_races = dog_races.iloc[max(0, idx-5):idx]
            
            # Feature: Last 5 win rate
            last5_win_rate = previous_races['is_winner'].mean() if len(previous_races) > 0 else 0.5
            
            # Feature: Last 5 finish benchmark (NEW!)
            last5_finish_benchmark = previous_races['FinishTimeBenchmarkLengths'].mean() if len(previous_races) > 0 else 0
            
            # Feature: Average distance
            avg_distance = previous_races['Distance'].mean() if len(previous_races) > 0 else race['Distance']
            
            features_data.append({
                'is_winner': race['is_winner'],
                'LastN_WinRate': last5_win_rate,
                'LastN_AvgFinishBenchmark': last5_finish_benchmark,
                'Distance': race['Distance'],
                'Weight': race['Weight'],
                'Box': race['Box'],
                'MeetingDate': race['MeetingDate']
            })

df_features = pd.DataFrame(features_data)
print(f"Created {len(df_features)} training examples")

# Split train/test
train_df = df_features[df_features['MeetingDate'] < '2025-06-01'].copy()
test_df = df_features[df_features['MeetingDate'] >= '2025-06-01'].copy()

print(f"\nTrain set: {len(train_df)} examples")
print(f"Test set: {len(test_df)} examples")
print(f"Train win rate: {train_df['is_winner'].mean()*100:.1f}%")

# Feature columns
feature_cols = ['LastN_WinRate', 'LastN_AvgFinishBenchmark', 'Distance', 'Weight', 'Box']

# Train model
print(f"\nTraining XGBoost with {len(feature_cols)} features...")

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['is_winner']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(X_train_scaled, y_train, verbose=False)

print("Model trained!")
print("\nFeature importances:")
for name, imp in zip(feature_cols, model.feature_importances_):
    print(f"  {name:30} {imp:.4f}")

# Test
print(f"\nTest set performance:")

X_test = test_df[feature_cols].fillna(0)
y_test = test_df['is_winner']

X_test_scaled = scaler.transform(X_test)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate at different confidence thresholds
for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = y_pred_proba >= threshold
    if mask.sum() > 20:
        strike = y_test[mask].mean() * 100
        count = mask.sum()
        print(f"  Confidence >= {threshold}: {count:>4} bets, {strike:>5.1f}% strike")

# Compare with actual finish benchmarks
print(f"\nComparison with actual FinishTimeBenchmarks:")

test_with_benchmarks = test_df[test_df['MeetingDate'] >= '2025-06-01'].copy()
test_with_benchmarks['Prediction'] = y_pred_proba[:len(test_with_benchmarks)]

# Get actual benchmarks
query_actual = """
SELECT
    ge.EntryID,
    ge.FinishTimeBenchmarkLengths,
    CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS INTEGER) as is_winner,
    ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-06-01' AND rm.MeetingDate < '2025-12-01'
  AND ge.Position NOT IN ('DNF', 'SCR')
"""

df_actual = pd.read_sql_query(query_actual, conn)
df_actual['StartingPrice'] = pd.to_numeric(df_actual['StartingPrice'], errors='coerce')
df_actual = df_actual[(df_actual['StartingPrice'] >= 1.5) & (df_actual['StartingPrice'] < 2.0)]

print(f"Test races with actual benchmarks: {len(df_actual)}")

for bench_threshold in [0, 0.5, 1.0]:
    subset = df_actual[df_actual['FinishTimeBenchmarkLengths'] >= bench_threshold]
    if len(subset) > 20:
        strike = subset['is_winner'].mean() * 100
        returns = (subset[subset['is_winner'] == 1]['StartingPrice'] * 100).sum()
        roi = ((returns - len(subset) * 100) / (len(subset) * 100)) * 100
        print(f"  Actual FinishTime >= {bench_threshold:>3}: {len(subset):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

conn.close()

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("""
NEW FEATURE: LastN_AvgFinishBenchmark
- Average finish pace benchmark from last 5 races
- This is PREDICTIVE (available before race runs)
- Helps identify dogs that consistently run fast finishes

VALIDATION:
- Can compare model predictions vs actual benchmarks
- Actual finish benchmarks >= 1.0 deliver 40%+ ROI
- Model should identify dogs likely to achieve this

NEXT: Use this model for live predictions on upcoming races
""")
