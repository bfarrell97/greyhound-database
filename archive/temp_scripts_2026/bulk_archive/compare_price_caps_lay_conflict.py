"""
Compare LAY-only conflict backtest results for PriceCap $15 vs $30
Uses BACK exclusion Steam_Prob < 0.20 and LAY thresholds [0.60,0.65,0.70,0.75]
Period: Jul-Dec 2025
"""
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sqlite3
import os
import sys
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41

START='2025-07-01'; END='2025-12-31'
BACK_EXCL=0.20
LAY_THRESH=[0.60,0.65,0.70,0.75,0.80]
PRICES=[15.0,30.0]

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

fe = FeatureEngineerV41()
df = fe.calculate_features(df)

# histories
df['MoveRatio'] = df['Price5Min'] / df['BSP']
df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)
df = df.sort_values('MeetingDate')
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
model_v41 = joblib.load('models/xgb_v41_final.pkl')
feats_v41 = fe.get_feature_list()
for c in feats_v41:
    if c not in df.columns: df[c]=0
ndf = df.copy()
ndf['V41_Prob'] = model_v41.predict(xgb.DMatrix(ndf[feats_v41]))
ndf['V41_Price'] = 1.0/ndf['V41_Prob']
ndf['Discrepancy'] = ndf['Price5Min']/ndf['V41_Price']
ndf['Price_Diff'] = ndf['Price5Min'] - ndf['V41_Price']

model_v44 = joblib.load('models/xgb_v44_production.pkl')
feats_v44 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Steam_10','Trainer_Rolling_Steam_50']
try:
    ndf['Steam_Prob'] = model_v44.predict_proba(ndf[feats_v44])[:,1]
except Exception:
    ndf['Steam_Prob'] = model_v44.get_booster().predict(xgb.DMatrix(ndf[feats_v44], feature_names=feats_v44))

model_v45 = joblib.load('models/xgb_v45_production.pkl')
feats_v45 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Drift_10','Trainer_Rolling_Drift_50']
try:
    ndf['Drift_Prob'] = model_v45.predict_proba(ndf[feats_v45])[:,1]
except Exception:
    ndf['Drift_Prob'] = model_v45.get_booster().predict(xgb.DMatrix(ndf[feats_v45], feature_names=feats_v45))

ndf['Dog_Win'] = (pd.to_numeric(ndf['Position'],errors='coerce')==1).astype(int)

rows=[]
for pc in PRICES:
    for lt in LAY_THRESH:
        mask = (ndf['Drift_Prob'] >= lt) & (ndf['Steam_Prob'] <= BACK_EXCL) & (ndf['Price5Min'] < pc)
        s = ndf[mask].copy()
        if len(s)==0:
            rows.append({'PriceCap':pc,'LayTh':lt,'Bets':0,'PnL':0.0,'SR':np.nan,'ROI':np.nan})
        else:
            s['PnL'] = np.where(s['Dog_Win']==0,1.0, -(s['Price5Min']-1.0))
            rows.append({'PriceCap':pc,'LayTh':lt,'Bets':len(s),'PnL':s['PnL'].sum(),'SR':(s['Dog_Win']==0).mean(),'ROI':s['PnL'].sum()/len(s)*100})

res=pd.DataFrame(rows)
print(res.pivot(index='LayTh',columns='PriceCap',values=['Bets','PnL','ROI']).to_string())
out='outputs/compare_pricecap_layconflict_2025H2.csv'
res.to_csv(out,index=False)
print(f'CSV saved to {out}')
