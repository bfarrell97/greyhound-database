"""
Backtest BACK-only strategy on Jul-Dec 2025
- BACK threshold: 0.40
- Price cap: $15
- Staking: target profit 4% of bankroll ($200 default)
Saves results to outputs/backtest_backonly_2025H2.csv
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
BACK_THRESH = 0.20
PRICE_CAP = 15.0
BANKROLL = 200.0
TARGET_PROFIT_PCT = 0.04

print(f"Loading data {START} -> {END}...")
conn = sqlite3.connect('greyhound_racing.db')
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

# V41 probs
print('Running V41...')
model_v41 = joblib.load('models/xgb_v41_final.pkl')
for c in features_v41:
    if c not in df.columns: df[c] = 0

dtest_v41 = xgb.DMatrix(df[features_v41])
df['V41_Prob'] = model_v41.predict(dtest_v41)

# STEAMER HISTORIES (ensure parity with grid backtest)
print('Engineering steamer history features...')
# MoveRatio and steamer flag
df['MoveRatio'] = df['Price5Min'] / df['BSP']
df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
# Sort and rolling
df = df.sort_values('MeetingDate')
df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(
    lambda x: x.rolling(window=10, min_periods=3).mean()
).fillna(0)

# Trainer-level steam
df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
    lambda x: x.rolling(window=50, min_periods=10).mean()
).fillna(0)

# Steam predictions
print('Running V44...')
features_v44 = [
    'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
    'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
    'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
]
# build discrepancy
df['V41_Price'] = 1.0 / df['V41_Prob']
df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
for c in features_v44:
    if c not in df.columns: df[c] = 0
model_v44 = joblib.load('models/xgb_v44_production.pkl')
X_v44 = df[features_v44].copy()
try:
    df['Steam_Prob'] = model_v44.predict_proba(X_v44)[:,1]
except Exception:
    dmat = xgb.DMatrix(X_v44, feature_names=features_v44)
    df['Steam_Prob'] = model_v44.get_booster().predict(dmat)

# Filter BACK-only
mask_back = (df['Steam_Prob'] >= BACK_THRESH) & (df['Price5Min'] < PRICE_CAP)
df_back = df[mask_back].copy()
print(f"BACK candidates: {len(df_back)}")

if df_back.empty:
    print('No BACK bets found with these criteria.')
    raise SystemExit

# Staking and PnL
target = BANKROLL * TARGET_PROFIT_PCT
stakes = []
for _, row in df_back.iterrows():
    price = row['Price5Min']
    if price <= 1.01:
        stake = 0.0
    else:
        stake = target / (price - 1.0)
    stakes.append(stake)

df_back['Stake'] = stakes
# risk is stake
# outcome
wins = (df_back['Position'] == 1)
# PnL: win -> stake*(price-1)*0.95 ; loss -> -stake
df_back['PnL'] = np.where(wins, df_back['Stake'] * (df_back['Price5Min'] - 1.0) * 0.95, -df_back['Stake'])

# Summary
total_bets = len(df_back)
win_rate = wins.mean()
total_pnl = df_back['PnL'].sum()
total_risk = df_back['Stake'].sum()
roi = (total_pnl / total_risk * 100) if total_risk > 0 else None
avg_stake = df_back['Stake'].mean()

print('\n=== BACK-only Backtest Results ===')
print(f'Threshold: {BACK_THRESH}, Price Cap: ${PRICE_CAP}')
print(f'Total Bets: {total_bets}')
print(f'Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)')
print(f'Total PnL: ${total_pnl:.2f}')
print(f'Total Stake (Risk): ${total_risk:.2f}')
print(f'ROI: {roi:.2f}%')
print(f'Avg Stake: ${avg_stake:.2f}')

# By month
if 'MeetingDate' in df_back.columns:
    df_back['MeetingDate'] = pd.to_datetime(df_back['MeetingDate'])
    monthly = df_back.groupby(df_back['MeetingDate'].dt.to_period('M')).agg({'PnL':'sum','Stake':'sum','EntryID':'count'})
    monthly['ROI_pct'] = monthly['PnL'] / monthly['Stake'] * 100
    print('\nMonthly breakdown:')
    print(monthly.to_string())

# Save bet-level CSV
out = os.path.join(OUT_DIR, 'backtest_backonly_2025H2.csv')
df_back.to_csv(out, index=False)
print(f'Bet-level CSV saved to {out}')
