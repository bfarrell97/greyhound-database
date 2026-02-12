"""
Grid backtest of V44 (BACK) and V45 (LAY) thresholds on last 6 months of 2025
Saves results to `outputs/grid_backtest_2025H2.csv` and prints top combos.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sys
sys.path.append('.')

from src.features.feature_engineering_v41 import FeatureEngineerV41

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Window: last 6 months of 2025
START = '2025-07-01'
END = '2025-12-31'
PRICE_CAP = 30.0

# Threshold ranges
BACK_THRESHOLDS = np.arange(0.30, 0.905, 0.05)  # 0.30..0.90
LAY_THRESHOLDS = np.arange(0.50, 0.805, 0.05)   # 0.50..0.80

print(f"Loading data {START} -> {END}...")
conn = sqlite3.connect('greyhound_racing.db')
query = f"""
SELECT 
    ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
    ge.Position, ge.FinishTime, ge.Split, 
    ge.BSP, ge.Price5Min, 
    ge.Weight, ge.Margin, ge.TrainerID,
    r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
    g.DateWhelped
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate >= '{START}'
AND rm.MeetingDate <= '{END}'
AND t.TrackName NOT IN ('LAUNCESTON', 'HOBART', 'DEVONPORT')
AND ge.Price5Min IS NOT NULL
AND ge.Price5Min > 0
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(df):,} rows")
if df.empty:
    raise SystemExit('No data for the period')

# Features
fe = FeatureEngineerV41()
df = fe.calculate_features(df)
features_v41 = fe.get_feature_list()

# Add steamer/drifter lag features similar to backtest_v44_thresholds
print("Engineering steam/drift history features...")
df['MoveRatio'] = df['Price5Min'] / df['BSP']
df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)

# Sort and create rolling
df = df.sort_values('MeetingDate')
df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(
    lambda x: x.rolling(window=10, min_periods=3).mean()
).fillna(0)

df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
    lambda x: x.rolling(window=50, min_periods=10).mean()
).fillna(0)

# DRIFT HISTORIES (for V45)
# Drifters are identified by MoveRatio < 0.95
df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)

# Dog-level drift
df['Prev_Drift'] = df.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
df['Dog_Rolling_Drift_10'] = df.groupby('GreyhoundID')['Prev_Drift'].transform(
    lambda x: x.rolling(window=10, min_periods=3).mean()
).fillna(0)

# Trainer-level drift
df['Trainer_Prev_Drift'] = df.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
df['Trainer_Rolling_Drift_50'] = df.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
    lambda x: x.rolling(window=50, min_periods=10).mean()
).fillna(0)

# V41 Prob
print("Generating V41 probabilities...")
model_v41 = joblib.load('models/xgb_v41_final.pkl')
for c in features_v41:
    if c not in df.columns: df[c] = 0

dtest_v41 = xgb.DMatrix(df[features_v41])
df['V41_Prob'] = model_v41.predict(dtest_v41)
df['V41_Price'] = 1.0 / df['V41_Prob']

# Discrepancy
df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
df['Price_Diff'] = df['Price5Min'] - df['V41_Price']

# V44/V45 predictions
features_v44 = [
    'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
    'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
    'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
]

print('Loading V44 production model...')
model_v44 = joblib.load('models/xgb_v44_production.pkl')
# Ensure all features exist
for c in features_v44:
    if c not in df.columns: df[c] = 0
X_v44 = df[features_v44].copy()
try:
    df['Steam_Prob'] = model_v44.predict_proba(X_v44)[:, 1]
except ValueError:
    # Fallback to Booster predict via DMatrix
    dmat = xgb.DMatrix(X_v44, feature_names=features_v44)
    try:
        df['Steam_Prob'] = model_v44.get_booster().predict(dmat)
    except Exception as e:
        print(f"Failed to predict Steam_Prob: {e}")
        raise

print('Loading V45 production model (fallback to 1-Steam if missing)...')
try:
    model_v45 = joblib.load('models/xgb_v45_production.pkl')
    # V45 expects drift history features (Dog_Rolling_Drift_10, Trainer_Rolling_Drift_50)
    features_v45 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
    ]
    for c in features_v45:
        if c not in df.columns: df[c] = 0
    X_v45 = df[features_v45].copy()
    try:
        df['Drift_Prob'] = model_v45.predict_proba(X_v45)[:, 1]
    except ValueError:
        dmat = xgb.DMatrix(X_v45, feature_names=features_v45)
        df['Drift_Prob'] = model_v45.get_booster().predict(dmat)
except Exception as e:
    print(f"V45 load failed: {e}; using 1 - Steam_Prob as proxy")
    df['Drift_Prob'] = 1 - df['Steam_Prob']

# Dog win flag
df['Dog_Win'] = (pd.to_numeric(df['Position'], errors='coerce') == 1).astype(int)

results = []
print('Starting grid sweep...')
for b in BACK_THRESHOLDS:
    for l in LAY_THRESHOLDS:
        # BACK control
        mask_back = (df['Steam_Prob'] >= b) & (df['Price5Min'] < PRICE_CAP)
        back = df[mask_back].copy()
        if len(back) > 0:
            back['PnL'] = np.where(back['Dog_Win'] == 1, (back['Price5Min'] - 1) * 0.95, -1)
            back_pnl = back['PnL'].sum()
            back_sr = back['Dog_Win'].mean()
            back_roi = back_pnl / len(back) * 100
        else:
            back_pnl = 0.0
            back_sr = np.nan
            back_roi = np.nan

        # LAY control
        mask_lay = (df['Drift_Prob'] >= l) & (df['Price5Min'] < PRICE_CAP)
        lay = df[mask_lay].copy()
        if len(lay) > 0:
            lay['PnL'] = np.where(lay['Dog_Win'] == 0, 1 * 0.95, -(lay['Price5Min'] - 1))
            lay_pnl = lay['PnL'].sum()
            lay_sr = (lay['Dog_Win'] == 0).mean()
            lay_roi = lay_pnl / len(lay) * 100
        else:
            lay_pnl = 0.0
            lay_sr = np.nan
            lay_roi = np.nan

        combined_pnl = back_pnl + lay_pnl
        combined_bets = len(back) + len(lay)
        combined_roi = combined_pnl / combined_bets * 100 if combined_bets > 0 else np.nan

        results.append({
            'BACK_thresh': round(b,3), 'LAY_thresh': round(l,3),
            'BACK_bets': len(back), 'BACK_pnl': back_pnl, 'BACK_SR': back_sr, 'BACK_ROI': back_roi,
            'LAY_bets': len(lay), 'LAY_pnl': lay_pnl, 'LAY_SR': lay_sr, 'LAY_ROI': lay_roi,
            'COMBINED_bets': combined_bets, 'COMBINED_pnl': combined_pnl, 'COMBINED_ROI': combined_roi
        })

res_df = pd.DataFrame(results)
res_csv = os.path.join(OUT_DIR, 'grid_backtest_2025H2.csv')
res_df.to_csv(res_csv, index=False)

print('\nTop 10 by COMBINED_ROI (min 20 bets):')
print(res_df[res_df['COMBINED_bets'] >= 20].sort_values('COMBINED_ROI', ascending=False).head(10).to_string(index=False))

print('\nTop 10 by COMBINED_PnL (min 20 bets):')
print(res_df[res_df['COMBINED_bets'] >= 20].sort_values('COMBINED_pnl', ascending=False).head(10).to_string(index=False))

print(f"\nResults saved to {res_csv}")

if __name__ == '__main__':
    pass
