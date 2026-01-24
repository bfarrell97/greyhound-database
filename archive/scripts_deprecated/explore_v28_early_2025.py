# explore_v28_early_2025.py
"""
Explore V28 model predictions for early 2025 races.
Loads the trained V28 AutoGluon model, applies the same preprocessing
as train_betfair_tutorial_v28.py, and generates predictions for races
in January 2025. Prints basic statistics: number of races, strike rate,
MAPE, correlation, and ROI (10% commission) for the early period.
"""
import sqlite3
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# ---------- 1. LOAD DATA (January 2025) ----------
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
    rm.MeetingDate as date_dt
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate < '2025-02-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('SCR', 'DNF', '')
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(df):,} entries for Jan 2025")

# ---------- 2. CLEANSE & NORMALISE (same as V28) ----------
df['date_dt'] = pd.to_datetime(df['date_dt'])
df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
df['SplitMargin'] = pd.to_numeric(df['SplitMargin'], errors='coerce')
df['StartPrice'] = pd.to_numeric(df['StartPrice'], errors='coerce')
df['Prizemoney'] = pd.to_numeric(df['Prizemoney'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)

df = df[df['Place'].notna()]
df = df[df['Box'] > 0]

df['win'] = (df['Place'] == 1).astype(int)
# Normalise start price probability per race
df['StartPrice_probability'] = (1 / df['StartPrice']).fillna(0)
df['StartPrice_probability'] = df.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x / x.sum())

# Normalise other columns (same as V28)
df['Prizemoney_norm'] = np.log10(df['Prizemoney'] + 1) / 12
df['Place_inv'] = (1 / df['Place']).fillna(0)
df['Place_log'] = np.log10(df['Place'] + 1).fillna(0)
df['RunSpeed'] = (df['RunTime'] / df['Distance']).fillna(0)

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

# ---------- 3. ROLLING WINDOW FEATURES ----------
features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm']
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
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability'] + feature_cols]

# ---------- 4. LOAD V28 MODEL ----------
predictor = TabularPredictor.load('models/autogluon_v28_tutorial')

# ---------- 5. PREDICTION & EVALUATION ----------
probs = predictor.predict_proba(model_df[feature_cols])
model_df = model_df.copy()
model_df['prob_model'] = probs[1]
# normalise per race
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# strike rate
races = model_df['FastTrack_RaceId'].nunique()
model_winners = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x == x.max())
model_strike = len(model_df[(model_winners) & (model_df['win'] == 1)]) / races

# price accuracy
valid = model_df[(model_df['StartPrice'] > 1) & (model_df['StartPrice'] < 50)].copy()
valid['RatedPrice'] = 1 / model_df['prob_model']
valid['PctError'] = np.abs(valid['RatedPrice'] - valid['StartPrice']) / valid['StartPrice']
MAPE = valid['PctError'].mean() * 100
corr = valid[['RatedPrice', 'StartPrice']].corr().iloc[0, 1]

# betting simulation (10% commission)
COMM = 0.10
bets = valid[(valid['StartPrice'] > valid['RatedPrice'] * 1.5) & (valid['StartPrice'] <= 10)].copy()
bets['Profit'] = np.where(bets['win'] == 1, (bets['StartPrice'] - 1) * (1 - COMM), -1)
roi = bets['Profit'].sum() / len(bets) * 100 if len(bets) > 0 else 0
wins = bets['win'].sum()
strike = wins / len(bets) * 100 if len(bets) > 0 else 0

print("=== V28 Early 2025 Evaluation ===")
print(f"Races: {races:,}")
print(f"Model Strike Rate: {model_strike:.2%}")
print(f"MAPE: {MAPE:.2f}%")
print(f"Correlation: {corr:.4f}")
print(f"Bets: {len(bets):,}, Wins: {wins:,} ({strike:.1f}%), ROI (after 10% commission): {roi:.2f}%")

