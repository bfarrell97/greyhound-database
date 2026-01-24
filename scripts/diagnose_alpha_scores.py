"""Diagnostic: Check V42/V43 Score Distribution"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41

conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
       ge.Position, ge.FinishTime, ge.Split, ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
       ge.Price5Min, r.Distance, t.TrackName, rm.MeetingDate, g.DateWhelped
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate >= '2024-06-01' AND ge.Price5Min > 0
LIMIT 500
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df)} rows")

fe = FeatureEngineerV41()
df = fe.calculate_features(df)

model_v41 = joblib.load('models/xgb_v41_final.pkl')
v41_cols = fe.get_feature_list()
for c in v41_cols:
    if c not in df.columns: df[c] = 0
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

dmatrix = xgb.DMatrix(df[v41_cols])
df['V41_Prob'] = model_v41.predict(dmatrix)
df['V41_Price'] = 1.0 / df['V41_Prob']
df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
df['Price_Diff'] = df['Price5Min'] - df['V41_Price']

alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']

m42 = joblib.load('models/xgb_v42_steamer.pkl')
m43 = joblib.load('models/xgb_v43_drifter.pkl')

df['V42'] = m42.predict_proba(df[alpha_feats])[:, 1]
df['V43'] = m43.predict_proba(df[alpha_feats])[:, 1]

print("\n=== V42 (Steamer) Score Distribution ===")
print(df['V42'].describe())
print(f"\nCount >= 0.60: {(df['V42'] >= 0.60).sum()}")
print(f"Count >= 0.53: {(df['V42'] >= 0.53).sum()}")
print(f"Count >= 0.50: {(df['V42'] >= 0.50).sum()}")

print("\n=== V43 (Drifter) Score Distribution ===")
print(df['V43'].describe())
print(f"\nCount >= 0.70: {(df['V43'] >= 0.70).sum()}")
print(f"Count >= 0.55: {(df['V43'] >= 0.55).sum()}")
print(f"Count >= 0.50: {(df['V43'] >= 0.50).sum()}")

print("\n=== Discrepancy (Ratio) Distribution ===")
print(df['Discrepancy'].describe())
print(f"\nCount Ratio > 1.05: {(df['Discrepancy'] > 1.05).sum()}")
print(f"Count Ratio > 1.20: {(df['Discrepancy'] > 1.20).sum()}")
