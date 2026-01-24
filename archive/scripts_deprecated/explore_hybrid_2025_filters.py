# explore_hybrid_2025_filters.py
"""
Hybrid V28/V30 evaluation on the full 2025 dataset with multiple betting filter
configurations. For each combination of value threshold, price cap, and optional
distance range, the script computes ROI, win‑rate and other metrics.
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
WHERE rm.MeetingDate >= '2023-01-01' AND rm.MeetingDate < '2026-01-01'
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('SCR', 'DNF', '')
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(df):,} entries for 2025")

# ---------- 2. CLEANSE & NORMALISE ----------
df['date_dt'] = pd.to_datetime(df['date_dt'])
numeric_cols = ['Place', 'RunTime', 'SplitMargin', 'StartPrice', 'Prizemoney', 'Distance', 'Box']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[df['Place'].notna()]
df = df[df['Box'] > 0]

df['win'] = (df['Place'] == 1).astype(int)
# Start price probability per race
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

# ---------- 3. EARLIEST PRICE ----------
price_order = ['Price2Hr', 'Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']
for col in price_order:
    if col not in df.columns:
        df[col] = np.nan

df['EarliestPrice'] = df[price_order].bfill(axis=1).iloc[:, 0]
df['EarliestPrice'].fillna(df['StartPrice'], inplace=True)

# ---------- 4. ROLLING FEATURES (including BSP_log) ----------
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
model_df = model_df[model_df['date_dt'] >= '2023-01-01']
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability', 'EarliestPrice', 'Distance'] + feature_cols]

# ---------- 5. LOAD MODELS ----------
predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial')
predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp')

# Ensure all required features exist
model_features = set(predictor_v28.feature_metadata.get_features()).union(predictor_v30.feature_metadata.get_features())
missing = model_features - set(feature_cols)
for col in missing:
    if col not in model_df.columns:
        model_df[col] = 0
        feature_cols.append(col)

# ---------- 6. PREDICTION ----------
probs_v28 = predictor_v28.predict_proba(model_df[feature_cols])
probs_v30 = predictor_v30.predict_proba(model_df[feature_cols])
model_df['prob_model'] = (probs_v28[1] + probs_v30[1]) / 2.0
# Normalise per race
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# ---------- 7. EXPERIMENTS ----------
value_thresholds = [0.75]
price_caps = [8]
# Example distance buckets (in metres) – limited to <550m
distance_bins = [(0, None)]

results = []
for vt in value_thresholds:
    for pc in price_caps:
        for dmin, dmax in distance_bins:
            # Filter by distance if bounds are set
            df_subset = model_df.copy()
            if dmin is not None:
                df_subset = df_subset[df_subset['Distance'] >= dmin]
            if dmax is not None:
                df_subset = df_subset[df_subset['Distance'] < dmax]
            # Compute rated price for this subset
            df_subset['RatedPrice'] = 1 / df_subset['prob_model']
            # Betting filter
            bets = df_subset[(df_subset['EarliestPrice'] > df_subset['RatedPrice'] * (1 + vt)) & (df_subset['EarliestPrice'] <= pc)].copy()
            if len(bets) == 0:
                continue
            bets['Profit'] = np.where(bets['win'] == 1, (bets['EarliestPrice'] - 1) * (1 - 0.10), -1)
            roi = bets['Profit'].sum() / len(bets) * 100
            win_rate = bets['win'].sum() / len(bets) * 100
            results.append({
                'value_threshold': vt,
                'price_cap': pc,
                'distance_range': f"{dmin}-{dmax if dmax else '∞'}",
                'bets': len(bets),
                'wins': bets['win'].sum(),
                'win_rate_%': round(win_rate, 1),
                'roi_%': round(roi, 2),
                'min_odds': round(bets['EarliestPrice'].min(), 2),
                'max_odds': round(bets['EarliestPrice'].max(), 2),
                'median_odds': round(bets['EarliestPrice'].median(), 2)
            })

print("=== Hybrid Betting Filter Experiments ===")
for r in results:
    print(f"Thresh={r['value_threshold']:.2f}, Cap=${r['price_cap']}, Dist={r['distance_range']}: Bets={r['bets']:,}, Wins={r['wins']:,} ({r['win_rate_%']}%), ROI={r['roi_%']}%, Odds[min]={r['min_odds']}, max={r['max_odds']}, median={r['median_odds']}")

# Example odds bands (in decimal odds)
odds_bins = [(0, 1.5), (1.5, 3), (3, 5), (5, 7), (7, 10), (10, float('inf'))]

# After printing overall results, compute ROI per odds band
print("=== ROI per Odds Band ===")
for low, high in odds_bins:
    # Ensure we are using the full set of potential bets for the odds band analysis,
    # not just the 'bets' from the last iteration of the main loop.
    # We need to re-apply the value threshold and minimum odds filter.
    # For simplicity, let's assume we want to analyze the odds bands for the *last*
    # value_threshold and distance_range combination processed in the main loop.
    # If a more general analysis across all combinations is needed, this section
    # would need to be integrated into the main loop or run separately for each combination.
    # For this change, we'll use the df_subset from the last iteration of the main loop.
    
    # Re-filter for the specific odds band and ensure it meets the value threshold and min odds
    band_bets = df_subset[(df_subset['EarliestPrice'] >= low) & 
                          (df_subset['EarliestPrice'] < high) &
                          (df_subset['EarliestPrice'] > df_subset['RatedPrice'] * (1 + vt)) & 
                          (df_subset['EarliestPrice'] >= 3)].copy()
    
    if len(band_bets) == 0:
        continue
    band_bets['Profit'] = np.where(band_bets['win'] == 1, (band_bets['EarliestPrice'] - 1) * (1 - 0.10), -1)
    band_roi = band_bets['Profit'].sum() / len(band_bets) * 100
    band_win_rate = band_bets['win'].sum() / len(band_bets) * 100
    print(f"Odds {low}-{high}: Bets={len(band_bets):,}, Wins={band_bets['win'].sum():,} ({round(band_win_rate,1)}%), ROI={round(band_roi,2)}%")

