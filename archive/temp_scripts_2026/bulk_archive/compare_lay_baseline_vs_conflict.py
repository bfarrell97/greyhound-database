"""
Compare LAY >= 0.60 stats with and without Steam_Prob < 0.30 exclusion
Period: Jul-Dec 2025, Price cap $30
"""
import sqlite3
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sys
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41

DB = 'greyhound_racing.db'
START='2025-07-01'
END='2025-12-31'
PRICE_CAP=30.0
LAY_TH = 0.60

# Load data
conn = sqlite3.connect(DB)
query = f"""
SELECT 
    ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
    ge.Position, ge.FinishTime, ge.Split, 
    ge.BSP, ge.Price5Min, 
    ge.Weight, ge.Margin, ge.TrainerID,
    r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
    g.DateWhelped, g.GreyhoundName as Dog
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

# steam/drift features

df['MoveRatio'] = df['Price5Min'] / df['BSP']
df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)
df = df.sort_values('MeetingDate')
df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(lambda x: x.rolling(window=10, min_periods=3).mean()).fillna(0)
df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(lambda x: x.rolling(window=50, min_periods=10).mean()).fillna(0)

# drift
ndf = df.copy()
ndf['Prev_Drift'] = ndf.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
ndf['Dog_Rolling_Drift_10'] = ndf.groupby('GreyhoundID')['Prev_Drift'].transform(lambda x: x.rolling(window=10, min_periods=3).mean()).fillna(0)
ndf['Trainer_Prev_Drift'] = ndf.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
ndf['Trainer_Rolling_Drift_50'] = ndf.groupby('TrainerID')['Trainer_Prev_Drift'].transform(lambda x: x.rolling(window=50, min_periods=10).mean()).fillna(0)

# Models
model_v41 = joblib.load('models/xgb_v41_final.pkl')
features_v41 = fe.get_feature_list()
for c in features_v41:
    if c not in ndf.columns: ndf[c]=0
ndf['V41_Prob'] = model_v41.predict(xgb.DMatrix(ndf[features_v41]))
ndf['V41_Price'] = 1.0/ndf['V41_Prob']
ndf['Discrepancy'] = ndf['Price5Min']/ndf['V41_Price']
ndf['Price_Diff'] = ndf['Price5Min'] - ndf['V41_Price']

# V44
features_v44 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Steam_10','Trainer_Rolling_Steam_50']
model_v44 = joblib.load('models/xgb_v44_production.pkl')
Xv44 = ndf[features_v44].copy()
try:
    ndf['Steam_Prob'] = model_v44.predict_proba(Xv44)[:,1]
except Exception:
    ndf['Steam_Prob'] = model_v44.get_booster().predict(xgb.DMatrix(Xv44, feature_names=features_v44))

# V45
features_v45 = ['Price5Min','V41_Prob','Discrepancy','Price_Diff','Box','Distance','RunTimeNorm_Lag1','Trainer_Track_Rate','Dog_Rolling_Drift_10','Trainer_Rolling_Drift_50']
model_v45 = joblib.load('models/xgb_v45_production.pkl')
Xv45 = ndf[features_v45].copy()
try:
    ndf['Drift_Prob'] = model_v45.predict_proba(Xv45)[:,1]
except Exception:
    ndf['Drift_Prob'] = model_v45.get_booster().predict(xgb.DMatrix(Xv45, feature_names=features_v45))

ndf['Dog_Win'] = (pd.to_numeric(ndf['Position'],errors='coerce')==1).astype(int)

# Baseline LAY >= 0.60, price cap
mask_base = (ndf['Drift_Prob'] >= LAY_TH) & (ndf['Price5Min'] < PRICE_CAP)
base = ndf[mask_base].copy()
if len(base) > 0:
    base['PnL'] = np.where(base['Dog_Win']==0, 1.0, -(base['Price5Min'] - 1.0))
    base_bets = len(base)
    base_pnl = base['PnL'].sum()
    base_sr = (base['Dog_Win']==0).mean()
    base_roi = base_pnl / base_bets * 100
else:
    base_bets=0; base_pnl=0; base_sr=np.nan; base_roi=np.nan

# Conflict LAY (Steam_Prob < 0.30)
mask_conf = mask_base & (ndf['Steam_Prob'] < 0.30)
conf = ndf[mask_conf].copy()
if len(conf) > 0:
    conf['PnL'] = np.where(conf['Dog_Win']==0, 1.0, -(conf['Price5Min'] - 1.0))
    conf_bets = len(conf)
    conf_pnl = conf['PnL'].sum()
    conf_sr = (conf['Dog_Win']==0).mean()
    conf_roi = conf_pnl / conf_bets * 100
else:
    conf_bets=0; conf_pnl=0; conf_sr=np.nan; conf_roi=np.nan

print('LAY >= 0.60 baseline: bets=%d | PnL=%.2f | SR=%.4f | ROI=%.2f%%' % (base_bets, base_pnl, base_sr, base_roi))
print('LAY >= 0.60 with Steam<0.30: bets=%d | PnL=%.2f | SR=%.4f | ROI=%.2f%%' % (conf_bets, conf_pnl, conf_sr, conf_roi))

# Additional check: any LAY bets with Steam_Prob >= 0.30?
mask_overlap = (ndf['Drift_Prob'] >= LAY_TH) & (ndf['Steam_Prob'] >= 0.30) & (ndf['Price5Min'] < PRICE_CAP)
overlap = ndf[mask_overlap]
print('\nOverlap (Drift>=0.60 AND Steam>=0.30): count=%d' % len(overlap))


