"""Validate PIR Leader + High WR + High Money strategy properly"""
import sqlite3
import pandas as pd
import numpy as np
import sys

def fp(msg):
    print(msg)
    sys.stdout.flush()

fp('='*70)
fp('VALIDATING PIR LEADER + HIGH WR + HIGH MONEY STRATEGY')
fp('='*70)
fp('')

conn = sqlite3.connect('greyhound_racing.db')

fp('Loading data...')
query = '''
SELECT 
    ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
    ge.Position, ge.StartingPrice, ge.CareerPrizeMoney,
    rm.MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.Position IS NOT NULL
  AND ge.StartingPrice IS NOT NULL
  AND ge.FirstSplitPosition IS NOT NULL
  AND ge.FirstSplitPosition != ''
  AND rm.MeetingDate >= '2020-01-01'
ORDER BY ge.GreyhoundID, rm.MeetingDate, r.RaceID
'''

df = pd.read_sql_query(query, conn)
fp(f'Loaded {len(df)} rows')

df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
df['Year'] = pd.to_datetime(df['MeetingDate']).dt.year

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
df['CumWins'] = df.groupby('GreyhoundID')['Win'].cumsum().shift(1).fillna(0)
df['HistWinRate'] = np.where(df['CumCount'] > 0, df['CumWins'] / df['CumCount'], 0)

box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1

# Strategy criteria
mask = (df['IsPIRLeader'] & 
        (df['HistWinRate'] >= 0.30) & 
        (df['CareerPrizeMoney'] >= 30000) &
        (df['StartingPrice'] >= 3.0) & 
        (df['StartingPrice'] <= 50))

subset = df[mask].copy()
fp(f'Strategy subset: {len(subset)} bets')

# Original ROI
bets = len(subset)
wins = subset['Win'].sum()
returns = (subset[subset['Win']]['StartingPrice'] - 1).sum()
orig_roi = (returns - bets) / bets * 100
orig_winrate = wins/bets*100

fp('')
fp(f'Our strategy: {bets} bets, {orig_winrate:.1f}% wins, ROI: {orig_roi:+.1f}%')
fp('')

# Get all dogs from races we bet on
bet_races = subset['RaceID'].unique()
all_in_races = df[df['RaceID'].isin(bet_races)].copy()

fp('='*70)
fp('BASELINE COMPARISONS')
fp('='*70)
fp('')

# Favorite baseline (in our bet races)
fav = all_in_races.loc[all_in_races.groupby('RaceID')['StartingPrice'].idxmin()]
fav_bets = len(fav)
fav_wins = fav['Win'].sum()
fav_returns = (fav[fav['Win']]['StartingPrice'] - 1).sum()
fav_roi = (fav_returns - fav_bets) / fav_bets * 100

fp(f'Favorite in same races: {fav_bets} bets, {fav_wins/fav_bets*100:.1f}% wins, ROI: {fav_roi:+.1f}%')
fp(f'Our strategy:           {bets} bets, {orig_winrate:.1f}% wins, ROI: {orig_roi:+.1f}%')
fp('')
fp(f'Improvement vs favorite betting: {orig_roi - fav_roi:+.1f}%')

# Same odds baseline (random dogs at $3-$50)
fp('')
fp('='*70)
fp('MONTE CARLO: Random dogs at same odds range')
fp('='*70)

same_odds = all_in_races[(all_in_races['StartingPrice'] >= 3) & (all_in_races['StartingPrice'] <= 50)]
fp(f'Pool: {len(same_odds)} dogs at 3-50')
fp('')

np.random.seed(42)
rand_rois = []
for i in range(100):
    sample = same_odds.sample(n=min(len(subset), len(same_odds)), replace=True)
    s_wins = sample['Win'].sum()
    s_returns = (sample[sample['Win']]['StartingPrice'] - 1).sum()
    s_roi = (s_returns - len(sample)) / len(sample) * 100
    rand_rois.append(s_roi)

avg_rand_roi = np.mean(rand_rois)
std_rand = np.std(rand_rois)

fp(f'Random 3-50 dog baseline: {avg_rand_roi:.1f}% ROI (avg of 100 samples)')
fp(f'Our strategy:             {orig_roi:+.1f}% ROI')
fp('')
fp(f'Edge vs random: {orig_roi - avg_rand_roi:+.1f}%')
fp('')

z = (orig_roi - avg_rand_roi) / std_rand if std_rand > 0 else 0
fp(f'Z-score: {z:.2f}')

if z > 3:
    fp('*** HIGHLY SIGNIFICANT (p < 0.001) ***')
elif z > 2:
    fp('*** STATISTICALLY SIGNIFICANT (p < 0.05) ***')
elif z > 1.5:
    fp('Marginally significant')
else:
    fp('Not statistically significant')

# Year by year
fp('')
fp('='*70)
fp('YEAR BY YEAR BREAKDOWN')
fp('='*70)

for year in sorted(subset['Year'].unique()):
    y = subset[subset['Year'] == year]
    bets_y = len(y)
    wins_y = y['Win'].sum()
    returns_y = (y[y['Win']]['StartingPrice'] - 1).sum()
    roi_y = (returns_y - bets_y) / bets_y * 100 if bets_y > 0 else 0
    fp(f'{year}: {bets_y} bets, {wins_y} wins ({wins_y/bets_y*100:.1f}%), ROI: {roi_y:+.1f}%')

# Odds breakdown
fp('')
fp('='*70)
fp('ODDS RANGE BREAKDOWN')  
fp('='*70)

for pmin, pmax in [(3, 5), (5, 10), (10, 20), (20, 50)]:
    s = subset[(subset['StartingPrice'] >= pmin) & (subset['StartingPrice'] < pmax)]
    if len(s) >= 10:
        bets_s = len(s)
        wins_s = s['Win'].sum()
        returns_s = (s[s['Win']]['StartingPrice'] - 1).sum()
        roi_s = (returns_s - bets_s) / bets_s * 100
        avg_odds = s['StartingPrice'].mean()
        fp(f'${pmin}-${pmax}: {bets_s} bets, {wins_s/bets_s*100:.1f}% wins, avg ${avg_odds:.1f}, ROI: {roi_s:+.1f}%')

# Daily volume
fp('')
fp('='*70)
fp('PRACTICAL METRICS')
fp('='*70)

days = len(subset['MeetingDate'].unique())
fp(f'Total betting days: {days}')
fp(f'Average bets per day: {len(subset)/days:.1f}')

conn.close()
fp('')
fp('VALIDATION COMPLETE')
