# explore_hybrid_2025_earliest.py
"""
Hybrid evaluation of V28 and V30 AutoGluon models on the full year 2025.
Uses the earliest available price column (Price2Hr, Price60Min, ..., Price1Min) with fallback to BSP.
Averages the win‑probability predictions from both models, normalises per race, and evaluates
price accuracy (MAPE, correlation) and a betting simulation (value > 50% and price ≤ $10, 10% commission).
"""
import sqlite3
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# ---------- 1. LOAD DATA (full 2025) ----------
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT
    ge.EntryID,
    ge.RaceID as FastTrack_RaceId,
    ge.GreyhoundID as FastTrack_DogId,
    ge.Box,
    ge.Weight,
    ge.Position as Place,
    ge.Margin as Margin1,
    ge.FinishTime as RunTime,
    ge.Split as SplitMargin,
    ge.BSP as StartPrice,
    ge.PrizeMoney as Prizemoney,
    r.Distance,
    t.TrackName as Track,
    rm.MeetingDate as date_dt,
    ge.Price2Hr,
    ge.Price60Min,
    ge.Price30Min,
    ge.Price15Min,
    ge.Price10Min,
    ge.Price5Min,
    ge.Price2Min,
    ge.Price1Min
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2026-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('SCR', 'DNF', '')
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(df):,} entries for 2025")

# ---------- 2. CLEANSE & NORMALISE (same as V28/V30) ----------
df['date_dt'] = pd.to_datetime(df['date_dt'])
numeric_cols = ['Place', 'RunTime', 'SplitMargin', 'StartPrice', 'Prizemoney', 'Distance', 'Box']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['Place'].notna()]
df = df[df['Box'] > 0]

df['win'] = (df['Place'] == 1).astype(int)
# Normalise start price probability per race
df['StartPrice_probability'] = (1 / df['StartPrice']).fillna(0)
df['StartPrice_probability'] = df.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x / x.sum())

# Additional normalisations
df['Prizemoney_norm'] = np.log10(df['Prizemoney'] + 1) / 12
df['Place_inv'] = (1 / df['Place']).fillna(0)
df['Place_log'] = np.log10(df['Place'] + 1).fillna(0)
df['RunSpeed'] = (df['RunTime'] / df['Distance']).fillna(0)

# Log‑transform BSP for V30 features
df['BSP_log'] = np.log(df['StartPrice'].clip(lower=1.01)).fillna(0)

# Track reference times
win_df = df[df['win'] == 1]
median_win_time = win_df[win_df['RunTime'] > 0].groupby(['Track', 'Distance'])['RunTime'].median().reset_index()
median_win_time.columns = ['Track', 'Distance', 'RunTime_median']
median_win_split = win_df[win_df['SplitMargin'] > 0].groupby(['Track', 'Distance'])['SplitMargin'].median().reset_index()
median_win_split.columns = ['Track', 'Distance', 'SplitMargin_median']
median_win_time['speed_index'] = median_win_time['RunTime_median'] / median_win_time['Distance']
median_win_time['speed_index'] = MinMaxScaler().fit_transform(median_win_time[['speed_index']])

df = df.merge(median_win_time[['Track', 'Distance', 'RunTime_median', 'speed_index']], on=['Track', 'Distance'], how='left')
df = df.merge(median_win_split, on=['Track', 'Distance'], how='left')

df['RunTime_norm'] = (df['RunTime_median'] / df['RunTime']).clip(0.9, 1.1)
df['RunTime_norm'] = MinMaxScaler().fit_transform(df[['RunTime_norm']])

df['SplitMargin_norm'] = (df['SplitMargin_median'] / df['SplitMargin']).clip(0.9, 1.1)
df['SplitMargin_norm'] = MinMaxScaler().fit_transform(df[['SplitMargin_norm']])

# Box win percentages
box_win = df.groupby(['Track', 'Distance', 'Box'])['win'].mean().reset_index()
box_win.columns = ['Track', 'Distance', 'Box', 'box_win_percent']
df = df.merge(box_win, on=['Track', 'Distance', 'Box'], how='left')

# ---------- 3. EARLIEST PRICE SELECTION ----------
price_order = ['Price2Hr', 'Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']
for col in price_order:
    if col not in df.columns:
        df[col] = np.nan
# Back‑fill across ordered columns to get earliest available price
df['EarliestPrice'] = df[price_order].bfill(axis=1).iloc[:, 0]
# Fallback to BSP if still missing
df['EarliestPrice'].fillna(df['StartPrice'], inplace=True)

# ---------- 4. ROLLING WINDOW FEATURES (including BSP_log) ----------
features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'BSP_log']
aggregates = ['min', 'max', 'mean', 'median', 'std']
rolling_windows = ['28D', '91D', '365D']

dataset = df.copy().set_index(['FastTrack_DogId', 'date_dt']).sort_index()
feature_cols = ['speed_index', 'box_win_percent']
for w in rolling_windows:
    rolling_res = (
        dataset.reset_index(level=0)
        .groupby('FastTrack_DogId')[features]
        .rolling(w)
        .agg(aggregates)
        .groupby(level=0)
        .shift(1)
    )
    agg_cols = [f"{f}_{a}_{w}" for f, a in itertools.product(features, aggregates)]
    dataset[agg_cols] = rolling_res
    feature_cols.extend(agg_cols)

dataset.fillna(0, inplace=True)
model_df = dataset.reset_index()
feature_cols = list(set(feature_cols))
model_df = model_df[model_df['date_dt'] >= '2025-01-01']
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability', 'EarliestPrice'] + feature_cols]

# ---------- 5. LOAD MODELS ----------
predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial')
predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp')

# ---------- 6. PREDICTION ----------
# Ensure all required features exist for both models
model_features_v28 = set(predictor_v28.feature_metadata.get_features())
model_features_v30 = set(predictor_v30.feature_metadata.get_features())
missing_v28 = model_features_v28 - set(feature_cols)
missing_v30 = model_features_v30 - set(feature_cols)
for col in missing_v28.union(missing_v30):
    if col not in model_df.columns:
        model_df[col] = 0
        feature_cols.append(col)

probs_v28 = predictor_v28.predict_proba(model_df[feature_cols])
probs_v30 = predictor_v30.predict_proba(model_df[feature_cols])
# Average win probability (class 1)
avg_prob = (probs_v28[1] + probs_v30[1]) / 2.0
model_df = model_df.copy()
model_df['prob_model'] = avg_prob
# Normalise per race so probabilities sum to 1
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# ---------- 7. METRICS ----------
races = model_df['FastTrack_RaceId'].nunique()
model_winners = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x == x.max())
model_strike = len(model_df[(model_winners) & (model_df['win'] == 1)]) / races

valid = model_df[(model_df['EarliestPrice'] > 1) & (model_df['EarliestPrice'] < 50)].copy()
valid['RatedPrice'] = 1 / model_df['prob_model']
valid['PctError'] = np.abs(valid['RatedPrice'] - valid['EarliestPrice']) / valid['EarliestPrice']
MAPE = valid['PctError'].mean() * 100
corr = valid[['RatedPrice', 'EarliestPrice']].corr().iloc[0, 1]

# Betting simulation (value > 50% and max $10)
COMM = 0.10
bets = valid[(valid['EarliestPrice'] > valid['RatedPrice'] * 1.5) & (valid['EarliestPrice'] <= 10)].copy()
bets['Profit'] = np.where(bets['win'] == 1, (bets['EarliestPrice'] - 1) * (1 - COMM), -1)
roi = bets['Profit'].sum() / len(bets) * 100 if len(bets) > 0 else 0
wins = bets['win'].sum()
strike = wins / len(bets) * 100 if len(bets) > 0 else 0

print("=== Hybrid V28/V30 2025 (Earliest Price) Evaluation ===")
print(f"Races: {races:,}")
print(f"Model Strike Rate: {model_strike:.2%}")
print(f"MAPE (earliest price): {MAPE:.2f}%")
print(f"Correlation (earliest price): {corr:.4f}")
print(f"Bets: {len(bets):,}, Wins: {wins:,} ({strike:.1f}%), ROI (after 10% commission): {roi:.2f}%")
