"""
Add historical finish pace feature to ML model training

This adds: avg_finish_benchmark_last_5_races
Which captures: How fast the dog runs relative to benchmark in recent races
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime, timedelta

DB_PATH = 'greyhound_racing.db'

print("="*100)
print("REBUILDING ML MODEL WITH FINISH PACE FEATURE")
print("="*100)

conn = sqlite3.connect(DB_PATH)

# Define training period
TRAIN_START = '2023-01-01'
TRAIN_END = '2025-05-31'

print(f"\nTraining period: {TRAIN_START} to {TRAIN_END}")

# Step 1: Build features with pace metrics
print("\nStep 1: Building feature dataset with pace metrics...")

query = """
WITH dog_features AS (
    SELECT
        ge.EntryID,
        ge.RaceID,
        ge.GreyhoundID,
        CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS REAL) as Label,
        -- Historical win rate (last 5 races)
        AVG(CAST((CASE WHEN LAG(ge.Position, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate) = '1' THEN 1 ELSE 0 END) AS REAL)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_WinRate,
        -- Historical average position (last 5 races)
        AVG(CAST(LAG(ge.Position, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate) AS REAL)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_AvgPosition,
        -- NEW: Historical finish pace benchmark (last 5 races)
        AVG(LAG(ge.FinishTimeBenchmarkLengths, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_AvgFinishBenchmark,
        -- Box win rate
        AVG(CAST((CASE WHEN LAG(ge.Position, 1) OVER (PARTITION BY ge.Box ORDER BY rm.MeetingDate) = '1' THEN 1 ELSE 0 END) AS REAL)) OVER (
            PARTITION BY ge.Box ORDER BY rm.MeetingDate 
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) as BoxWinRate,
        ge.Weight,
        r.Distance,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
)
SELECT * FROM dog_features
"""

df_train = pd.read_sql_query(query, conn, params=(TRAIN_START, TRAIN_END))

print(f"  Loaded {len(df_train)} training examples")
print(f"  Features with LastN_AvgFinishBenchmark: {df_train['LastN_AvgFinishBenchmark'].notna().sum()}")

# Step 2: Prepare features
print("\nStep 2: Preparing feature matrix...")

# Columns to use as features
feature_cols = [
    'LastN_WinRate',
    'LastN_AvgPosition', 
    'LastN_AvgFinishBenchmark',  # NEW FEATURE!
    'BoxWinRate',
    'Weight',
    'Distance'
]

# Fill missing values
df_train_clean = df_train[feature_cols + ['Label']].dropna()

print(f"  Clean dataset: {len(df_train_clean)} examples")
print(f"  Win rate in training: {df_train_clean['Label'].mean()*100:.1f}%")

# Step 3: Train model
print("\nStep 3: Training XGBoost model...")

X = df_train_clean[feature_cols].values
y = df_train_clean['Label'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_scaled, y, verbose=False)

print(f"  Model trained successfully")
print(f"  Feature importances:")
for name, importance in zip(feature_cols, model.feature_importances_):
    print(f"    {name:30} {importance:.4f}")

# Step 4: Test on validation period
print("\nStep 4: Testing on validation period (2025-06-01 to 2025-11-30)...")

TEST_START = '2025-06-01'
TEST_END = '2025-12-01'

query_test = """
WITH dog_features AS (
    SELECT
        ge.EntryID,
        ge.RaceID,
        ge.GreyhoundID,
        CAST((CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) AS REAL) as Label,
        AVG(CAST((CASE WHEN LAG(ge.Position, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate) = '1' THEN 1 ELSE 0 END) AS REAL)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_WinRate,
        AVG(CAST(LAG(ge.Position, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate) AS REAL)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_AvgPosition,
        AVG(LAG(ge.FinishTimeBenchmarkLengths, 1) OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate)) OVER (
            PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) as LastN_AvgFinishBenchmark,
        AVG(CAST((CASE WHEN LAG(ge.Position, 1) OVER (PARTITION BY ge.Box ORDER BY rm.MeetingDate) = '1' THEN 1 ELSE 0 END) AS REAL)) OVER (
            PARTITION BY ge.Box ORDER BY rm.MeetingDate 
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) as BoxWinRate,
        ge.Weight,
        r.Distance,
        ge.StartingPrice,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR')
)
SELECT * FROM dog_features
"""

df_test = pd.read_sql_query(query_test, conn, params=(TEST_START, TEST_END))

# Filter to $1.50-$2.00 odds (our profitable range)
df_test['StartingPrice'] = pd.to_numeric(df_test['StartingPrice'], errors='coerce')
df_test = df_test[(df_test['StartingPrice'] >= 1.5) & (df_test['StartingPrice'] < 2.0)]

df_test_clean = df_test[feature_cols + ['Label', 'StartingPrice']].dropna()

print(f"  Test set: {len(df_test_clean)} examples at $1.50-$2.00")

# Make predictions
X_test_scaled = scaler.transform(df_test_clean[feature_cols].values)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

df_test_clean['Prediction'] = y_pred_proba
df_test_clean['Predicted_Win'] = y_pred

# Evaluate by prediction confidence
print("\n  Performance by model confidence:")
for conf_thresh in [0.5, 0.6, 0.7, 0.8]:
    subset = df_test_clean[df_test_clean['Prediction'] >= conf_thresh]
    if len(subset) > 20:
        strike = subset['Label'].mean() * 100
        
        stake_pct = 0.02
        bankroll = 1000
        total_staked = len(subset) * bankroll * stake_pct
        returns = (subset[subset['Label'] == 1]['StartingPrice'] * bankroll * stake_pct).sum()
        roi = ((returns - total_staked) / total_staked) * 100
        
        print(f"    Confidence >= {conf_thresh}: {len(subset):>4} bets, {strike:>5.1f}% strike, ROI {roi:>6.2f}%")

conn.close()

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)
print("""
NEW FEATURE ADDED: LastN_AvgFinishBenchmark
- Measures dog's historical finish pace relative to benchmark
- Based on last 5 races before the current race
- PREDICTIVE: Available before race runs
- Captures dogs that run consistently good pace

Next: Compare this model vs baseline to see if it improves predictions.
""")
