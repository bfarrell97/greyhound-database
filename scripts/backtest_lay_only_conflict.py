"""
LAY-only backtest where LAY threshold > production LAY (0.55)
and BACK (Steam_Prob) < 0.30 (conflict exclusion)
Period: Jul-Dec 2025
Price cap: $30
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

START = '2025-07-01'
END = '2025-12-31'
PRICE_CAP = 15.0
BACK_EXCLUDE_TH = 0.20  # Steam_Prob < 0.20
LAY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

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

# Steamer/Drift history
print('Engineering steam/drift histories...')
df['MoveRatio'] = df['Price5Min'] / df['BSP']
df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)

df = df.sort_values('MeetingDate')
# steam
df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(lambda x: x.rolling(window=10, min_periods=3).mean()).fillna(0)
df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(lambda x: x.rolling(window=50, min_periods=10).mean()).fillna(0)
# drift
df['Prev_Drift'] = df.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
df['Dog_Rolling_Drift_10'] = df.groupby('GreyhoundID')['Prev_Drift'].transform(lambda x: x.rolling(window=10, min_periods=3).mean()).fillna(0)
df['Trainer_Prev_Drift'] = df.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
df['Trainer_Rolling_Drift_50'] = df.groupby('TrainerID')['Trainer_Prev_Drift'].transform(lambda x: x.rolling(window=50, min_periods=10).mean()).fillna(0)

# V41
print('Generating V41 probabilities...')
model_v41 = joblib.load('models/xgb_v41_final.pkl')
for c in features_v41:
    if c not in df.columns: df[c] = 0
dtest_v41 = xgb.DMatrix(df[features_v41])
df['V41_Prob'] = model_v41.predict(dtest_v41)

df['V41_Price'] = 1.0 / df['V41_Prob']
df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
df['Price_Diff'] = df['Price5Min'] - df['V41_Price']

# V44 steam
features_v44 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Steam_10','Trainer_Rolling_Steam_50']
model_v44 = joblib.load('models/xgb_v44_production.pkl')
X_v44 = df[features_v44].copy()
try:
    df['Steam_Prob'] = model_v44.predict_proba(X_v44)[:,1]
except Exception:
    dmat = xgb.DMatrix(X_v44, feature_names=features_v44)
    df['Steam_Prob'] = model_v44.get_booster().predict(dmat)

# V45 drift
features_v45 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Drift_10','Trainer_Rolling_Drift_50']
model_v45 = joblib.load('models/xgb_v45_production.pkl')
X_v45 = df[features_v45].copy()
try:
    df['Drift_Prob'] = model_v45.predict_proba(X_v45)[:,1]
except Exception:
    dmat = xgb.DMatrix(X_v45, feature_names=features_v45)
    df['Drift_Prob'] = model_v45.get_booster().predict(dmat)

# Dog win flag
df['Dog_Win'] = (pd.to_numeric(df['Position'], errors='coerce') == 1).astype(int)

results = []
print('Testing LAY thresholds > production (0.55) with Steam_Prob < 0.30')
for lt in LAY_THRESHOLDS:
    mask = (df['Drift_Prob'] >= lt) & (df['Steam_Prob'] < BACK_EXCLUDE_TH) & (df['Price5Min'] < PRICE_CAP)
    lay = df[mask].copy()
    if len(lay) == 0:
        lay_bets = 0
        lay_pnl = 0.0
        lay_sr = np.nan
        lay_roi = np.nan
    else:
        # unit liability 1: win when Dog_Win==0 -> +1 ; lose when Dog_Win==1 -> -(Price-1)
        lay['PnL'] = np.where(lay['Dog_Win'] == 0, 1.0, -(lay['Price5Min'] - 1.0))
        lay_bets = len(lay)
        lay_pnl = lay['PnL'].sum()
        lay_sr = (lay['Dog_Win'] == 0).mean()
        lay_roi = lay_pnl / lay_bets * 100

    results.append({'LAY_thresh': lt, 'LAY_bets': lay_bets, 'LAY_pnl': lay_pnl, 'LAY_SR': lay_sr, 'LAY_ROI': lay_roi})

res_df = pd.DataFrame(results)
res_csv = os.path.join(OUT_DIR, 'lay_only_conflict_2025H2.csv')
res_df.to_csv(res_csv, index=False)

print('\nLAY-only results:')
print(res_df.to_string(index=False))
print(f'CSV saved to {res_csv}')
