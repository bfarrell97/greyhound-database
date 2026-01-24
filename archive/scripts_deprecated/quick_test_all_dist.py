"""Quick test: Gap > 0.15, Prize > 20k, Odds 2-30 - ALL DISTANCES"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

conn = sqlite3.connect(DB_PATH)
query = """
SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
       ge.FinishTime, ge.Position, ge.StartingPrice, ge.BSP, COALESCE(ge.PrizeMoney, 0) as PrizeMoney
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2024-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)

def parse_price(x):
    try:
        if not x: return np.nan
        return float(str(x).replace('$','').replace('F','').strip())
    except: return np.nan
    
df['SP'] = df['StartingPrice'].apply(parse_price)
df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')

pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']

df = df.sort_values(['GreyhoundID', 'MeetingDate'])
g = df.groupby('GreyhoundID')
df['p_Lag1'] = g['NormTime'].shift(1)
df['p_Lag2'] = g['NormTime'].shift(2)
df['p_Lag3'] = g['NormTime'].shift(3)
df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
df['PrevDate'] = g['MeetingDate'].shift(1)
df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)

df = df.dropna(subset=['p_Roll5']).copy()

with open(PACE_MODEL_PATH, 'rb') as f: model = pickle.load(f)
X = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
X.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
df['PredPace'] = model.predict(X) + df['TrackDistMedianPace']

df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
df['FieldSize'] = df.groupby('RaceKey')['SP'].transform('count')
df = df[df['FieldSize'] >= 6]

df = df.sort_values(['RaceKey', 'PredPace'])
df['Rank'] = df.groupby('RaceKey').cumcount() + 1
df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
df['Gap'] = df['NextTime'] - df['PredPace']

leaders = df[df['Rank'] == 1].copy()

# No distance filter - Gap > 0.15, Prize > 20k, Odds 2-30
filt_bsp = leaders[(leaders['Gap'] >= 0.15) & (leaders['CareerPrize'] >= 20000) & (leaders['BSP'] >= 2) & (leaders['BSP'] <= 30) & leaders['BSP'].notna()]
filt_sp = leaders[(leaders['Gap'] >= 0.15) & (leaders['CareerPrize'] >= 20000) & (leaders['SP'] >= 2) & (leaders['SP'] <= 30) & leaders['SP'].notna()]

filt_bsp['Profit'] = filt_bsp.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
filt_sp['Profit'] = filt_sp.apply(lambda x: (x['SP'] - 1) if x['Position'] == '1' else -1, axis=1)

print('='*70)
print('DOMINANT LEADER (Gap > 0.15s, Prize > 20k, Odds 2-30) - ALL DISTANCES')
print('='*70)
print(f"{'Price':<6} | {'Bets':<6} | {'Wins':<6} | {'Strike %':<9} | {'Profit':<10} | {'ROI %':<8}")
print('-'*70)

bsp_wins = filt_bsp[filt_bsp['Position'] == '1'].shape[0]
print(f"{'BSP':<6} | {len(filt_bsp):<6} | {bsp_wins:<6} | {(bsp_wins/len(filt_bsp))*100:<9.1f} | {filt_bsp['Profit'].sum():<10.1f} | {(filt_bsp['Profit'].sum()/len(filt_bsp))*100:<8.1f}")

sp_wins = filt_sp[filt_sp['Position'] == '1'].shape[0]
print(f"{'SP':<6} | {len(filt_sp):<6} | {sp_wins:<6} | {(sp_wins/len(filt_sp))*100:<9.1f} | {filt_sp['Profit'].sum():<10.1f} | {(filt_sp['Profit'].sum()/len(filt_sp))*100:<8.1f}")

print(f"\nAvg BSP: ${filt_bsp['BSP'].mean():.2f}, Avg SP: ${filt_sp['SP'].mean():.2f}")
