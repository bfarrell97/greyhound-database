# explore_hybrid_drawdown.py
"""
Calculates the Max Drawdown and Equity Curve for the Hybrid V28/V30 strategy
on the 2023-2025 dataset.
Filters: Value Threshold = 0.75, Price Cap = $8.
Staking: Flat $1 bet.
"""

import sqlite3
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# ---------- 1. LOAD DATA (2023‑2025) ----------
print("Loading data...")
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
WHERE rm.MeetingDate >= '2024-01-01' AND rm.MeetingDate < '2026-01-01'
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

df = df[df['Place'].notna()]
df = df[df['Box'] > 0]
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

print("Calculating rolling features...")
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

# Filter columns
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability', 'EarliestPrice', 'Distance'] + feature_cols]

# ---------- 5. LOAD PRE‑TRAINED MODELS ----------
print("Loading models...")
predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial')
predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp')

# Hybrid prediction
print("Predicting probabilities...")
prob_v28 = predictor_v28.predict_proba(model_df)
prob_v30 = predictor_v30.predict_proba(model_df)

prob_hybrid = (prob_v28[1] + prob_v30[1]) / 2 
model_df['prob_model'] = prob_hybrid
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# ---------- 6. BETTING FILTER & DRAWDOWN ----------
value_threshold = 0.75  # 75 % value
price_cap = 8           # $8 cap

print("Applying betting filter...")
model_df['RatedPrice'] = 1 / model_df['prob_model']

# Select bets
bets = model_df[(model_df['EarliestPrice'] > model_df['RatedPrice'] * (1 + value_threshold)) &
                (model_df['EarliestPrice'] <= price_cap)].copy()

# Sort by date to calculate running equity
bets = bets.sort_values('date_dt')

if len(bets) == 0:
    print("No bets found with current filters.")
else:
    # Calculate profit per bet ($1 flat stake)
    bets['Profit'] = np.where(bets['win'] == 1,
                              (bets['EarliestPrice'] - 1) * (1 - 0.10),
                              -1)

    bets['CumulativeProfit'] = bets['Profit'].cumsum()
    
    # Calculate Drawdown
    bets['HighWaterMark'] = bets['CumulativeProfit'].cummax()
    bets['Drawdown'] = bets['CumulativeProfit'] - bets['HighWaterMark']
    
    max_drawdown = bets['Drawdown'].min()
    current_profit = bets['CumulativeProfit'].iloc[-1]
    total_bets = len(bets)
    win_rate = bets['win'].mean() * 100
    roi = bets['Profit'].sum() / total_bets * 100
    
    # Calculate Max Drawdown Duration (consecutive bets in drawdown)
    # Identify when we are in drawdown
    in_drawdown = bets['Drawdown'] < 0
    # Group consecutive True values
    drawdown_periods = in_drawdown.ne(in_drawdown.shift()).cumsum()
    # Count lengths of these periods where in_drawdown is True
    drawdown_lengths = drawdown_periods[in_drawdown].value_counts()
    max_drawdown_duration = drawdown_lengths.max() if not drawdown_lengths.empty else 0

    print("\n=== Strategy Performance (2023-2025) ===")
    print(f"Filters: Value > {value_threshold}, Price < ${price_cap}")
    print(f"Total Bets: {total_bets:,}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${current_profit:,.2f} (units)")
    print(f"ROI: {roi:.2f}%")
    print("-" * 30)
    print(f"Max Drawdown: ${max_drawdown:,.2f} (units)")
    print(f"Max Drawdown Duration: {max_drawdown_duration} bets")
    
    # Additional stats
    print(f"Average Odds: ${bets['EarliestPrice'].mean():.2f}")
    print(f"Median Odds: ${bets['EarliestPrice'].median():.2f}")
    
    # Year by Year breakdown
    bets['Year'] = bets['date_dt'].dt.year
    print("\n=== Yearly Breakdown ===")
    yearly = bets.groupby('Year').agg({
        'Profit': 'sum',
        'win': 'count'
    }).rename(columns={'win': 'Bets', 'Profit': 'Profit'})
    yearly['ROI'] = (yearly['Profit'] / yearly['Bets'] * 100).round(2)
    print(yearly)

