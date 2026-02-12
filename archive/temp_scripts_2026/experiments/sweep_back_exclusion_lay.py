"""
Sweep back-exclusion thresholds and evaluate LAY performance
Back exclusion: Steam_Prob < back_excl (0.10..0.30 step 0.05)
LAY thresholds: [0.60,0.65,0.70,0.75,0.80]
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

OUT = 'outputs/sweep_back_exclusion_lay_2025H2.csv'
START='2025-07-01'
END='2025-12-31'
PRICE_CAP=30.0
BACK_EXCL = np.arange(0.10, 0.305, 0.05)  # 0.10,0.15,0.20,0.25,0.30
LAY_THRESH = [0.60,0.65,0.70,0.75,0.80]

print('Loading data...')
conn = sqlite3.connect('greyhound_racing.db')
query = f"""
SELECT ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, ge.Position, ge.FinishTime, ge.Split, ge.BSP, ge.Price5Min,
       ge.Weight, ge.Margin, ge.TrainerID, r.Distance, r.Grade, t.TrackName, rm.MeetingDate, g.DateWhelped
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate >= '{START}' AND rm.MeetingDate <= '{END}'
AND ge.Price5Min IS NOT NULL AND ge.Price5Min > 0
AND t.TrackName NOT IN ('LAUNCESTON','HOBART','DEVONPORT')
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f'Loaded {len(df):,} rows')

fe = FeatureEngineerV41()
df = fe.calculate_features(df)

# histories
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

# models
print('Loading models...')
model_v41 = joblib.load('models/xgb_v41_final.pkl')
feats_v41 = fe.get_feature_list()
for c in feats_v41:
    if c not in df.columns: df[c]=0
df['V41_Prob'] = model_v41.predict(xgb.DMatrix(df[feats_v41]))
df['V41_Price'] = 1.0/df['V41_Prob']
df['Discrepancy'] = df['Price5Min']/df['V41_Price']
df['Price_Diff'] = df['Price5Min'] - df['V41_Price']

model_v44 = joblib.load('models/xgb_v44_production.pkl')
feats_v44 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Steam_10','Trainer_Rolling_Steam_50']
X44 = df[feats_v44].copy()
try:
    df['Steam_Prob'] = model_v44.predict_proba(X44)[:,1]
except Exception:
    df['Steam_Prob'] = model_v44.get_booster().predict(xgb.DMatrix(X44, feature_names=feats_v44))

model_v45 = joblib.load('models/xgb_v45_production.pkl')
feats_v45 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Drift_10','Trainer_Rolling_Drift_50']
X45 = df[feats_v45].copy()
try:
    df['Drift_Prob'] = model_v45.predict_proba(X45)[:,1]
except Exception:
    df['Drift_Prob'] = model_v45.get_booster().predict(xgb.DMatrix(X45, feature_names=feats_v45))

# evaluate
results=[]
for be in BACK_EXCL:
    for lt in LAY_THRESH:
        mask = (df['Drift_Prob'] >= lt) & (df['Steam_Prob'] < be) & (df['Price5Min'] < PRICE_CAP)
        subset = df[mask].copy()
        if len(subset)==0:
            bets=0; pnl=0.0; sr=np.nan; roi=np.nan
        else:
            subset['PnL'] = np.where(subset['Position']==1, -(subset['Price5Min']-1.0), 1.0)
            bets = len(subset)
            pnl = subset['PnL'].sum()
            sr = (subset['Position'] != 1).mean()
            roi = pnl / bets * 100
        results.append({'Back_Excl': round(be,2), 'Lay_Th': round(lt,2), 'Bets': bets, 'PnL': round(pnl,2), 'SR': round(sr,4) if not np.isnan(sr) else np.nan, 'ROI_pct': round(roi,4) if not np.isnan(roi) else np.nan})

res_df = pd.DataFrame(results)
res_df.to_csv(OUT, index=False)
print('\nTop combos by PnL:')
print(res_df[res_df['Bets']>0].sort_values('PnL', ascending=False).head(10).to_string(index=False))
print(f'Full results saved to {OUT}')
