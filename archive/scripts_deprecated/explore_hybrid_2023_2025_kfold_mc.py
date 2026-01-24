# explore_hybrid_2023_2025_kfold_mc.py
"""
Hybrid V28/V30 evaluation with cross‑validation.
- Loads the pre‑trained V28 and V30 models.
- Replicates exact feature engineering from `explore_hybrid_2025_filters.py`.
- Performs 5‑fold K‑Fold evaluation and Monte‑Carlo (random train‑test) evaluation.
- For each split it applies the betting filter (value threshold 0.75, price cap $8, all distances) and computes ROI.
- Prints per‑fold ROI and overall mean/std for both methods.
"""

import sqlite3
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from autogluon.tabular import TabularPredictor

# ---------- 1. LOAD DATA (2023‑2025) ----------
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
WHERE rm.MeetingDate >= '2023-01-01' AND rm.MeetingDate < '2026-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('SCR', 'DNF', '')
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(df):,} entries for 2023‑2025")

# ---------- 2. CLEANSE & NORMALISE ----------
df['date_dt'] = pd.to_datetime(df['date_dt'])
numeric_cols = ['Place', 'RunTime', 'SplitMargin', 'StartPrice', 'Prizemoney', 'Distance', 'Box']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep only valid rows
df = df[df['Place'].notna()]
df = df[df['Box'] > 0]

# Target variable
df['win'] = (df['Place'] == 1).astype(int)

# Start price probability per race
df['StartPrice_probability'] = (1 / df['StartPrice']).fillna(0)
df['StartPrice_probability'] = df.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x / x.sum())

# Additional features
df['Prizemoney_norm'] = np.log10(df['Prizemoney'] + 1) / 12
df['Place_inv'] = (1 / df['Place']).fillna(0)
df['Place_log'] = np.log10(df['Place'] + 1).fillna(0)
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

# ---------- 3. EARLIEST PRICE ----------
price_order = ['Price2Hr', 'Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']
for col in price_order:
    if col not in df.columns:
        df[col] = np.nan

df['EarliestPrice'] = df[price_order].bfill(axis=1).iloc[:, 0]
df['EarliestPrice'].fillna(df['StartPrice'], inplace=True)

# ---------- 4. ROLLING FEATURES (Exact replication) ----------
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

# Filter to desired date range for EVALUATION (we already loaded 2023-2025)
# But we must ensure specific columns are present
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability', 'EarliestPrice', 'Distance'] + feature_cols]

# ---------- 5. LOAD PRE‑TRAINED MODELS ----------
print("Loading models...")
predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial')
predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp')

# Hybrid prediction (average probabilities)
print("Predicting V28...")
prob_v28 = predictor_v28.predict_proba(model_df)
print("Predicting V30...")
prob_v30 = predictor_v30.predict_proba(model_df)

# Assuming the positive class is column '1'
prob_hybrid = (prob_v28[1] + prob_v30[1]) / 2 # Use integer 1 directly or investigate column names if string '1'
model_df['prob_model'] = prob_hybrid
# Normalise per race
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# ---------- 6. BETTING FILTER ----------
value_threshold = 0.75  # 75 % value
price_cap = 8           # $8 cap
min_odds = 0            # No lower bound for now, or match previous script (which had no lower bound at end)
                        # Previous request was $1.5-$3 band, but main loop was 'all'. 
                        # Let's stick to the $8 cap and no lower bound.

def evaluate(df_subset: pd.DataFrame) -> dict:
    """Apply betting filter and compute ROI metrics for a given subset."""
    df_subset = df_subset.copy()
    df_subset['RatedPrice'] = 1 / df_subset['prob_model']
    bets = df_subset[(df_subset['EarliestPrice'] > df_subset['RatedPrice'] * (1 + value_threshold)) &
                    (df_subset['EarliestPrice'] <= price_cap)].copy()
    
    if len(bets) == 0:
        return {'bets': 0, 'wins': 0, 'roi': 0.0, 'win_rate': 0.0}
        
    bets['Profit'] = np.where(bets['win'] == 1,
                              (bets['EarliestPrice'] - 1) * (1 - 0.10),
                              -1)
    profit = bets['Profit'].sum()
    roi = profit / len(bets) * 100
    win_rate = bets['win'].sum() / len(bets) * 100
    return {'bets': len(bets), 'wins': bets['win'].sum(), 'roi': roi, 'win_rate': win_rate}

# ---------- 7. K‑FOLD EVALUATION ----------
print("\n=== K‑Fold Evaluation (5 folds) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Use FastTrack_RaceId for splitting to prevent data leakage (all dogs in a race should be in same fold)
unique_races = model_df['FastTrack_RaceId'].unique()

for fold, (train_race_idx, test_race_idx) in enumerate(kf.split(unique_races), start=1):
    test_races = unique_races[test_race_idx]
    test_df = model_df[model_df['FastTrack_RaceId'].isin(test_races)]
    
    res = evaluate(test_df)
    fold_results.append(res)
    print(f"Fold {fold}: Bets={res['bets']:,}, Wins={res['wins']:,} ({res['win_rate']:.1f}%), ROI={res['roi']:.2f}%")

kf_roi = [r['roi'] for r in fold_results]
print(f"K‑Fold ROI – mean: {np.mean(kf_roi):.2f}%, std: {np.std(kf_roi):.2f}%")

# ---------- 8. MONTE‑CARLO (random splits) ----------
print("\n=== Monte‑Carlo Evaluation (30 random splits) ===")
mc_iterations = 30
mc_results = []

for i in range(1, mc_iterations + 1):
    # Split by races again
    train_races, test_races = train_test_split(unique_races, test_size=0.3, random_state=i)
    test_df = model_df[model_df['FastTrack_RaceId'].isin(test_races)]
    
    res = evaluate(test_df)
    mc_results.append(res)
    # Print every 5th iteration to reduce clutter, or all if preferred
    if i % 1 == 0:
        print(f"Iter {i:02d}: Bets={res['bets']:,}, Wins={res['wins']:,} ({res['win_rate']:.1f}%), ROI={res['roi']:.2f}%")

mc_roi = [r['roi'] for r in mc_results]
print(f"Monte‑Carlo ROI – mean: {np.mean(mc_roi):.2f}%, std: {np.std(mc_roi):.2f}%")
