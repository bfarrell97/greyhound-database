"""
Backtest PIR + Pace strategies for 2025 with detailed CSV output
Exports all bets with full information for analysis
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

print('='*80)
print('BACKTEST 2025 - DETAILED CSV EXPORT')
print('='*80)
print('')

conn = sqlite3.connect('greyhound_racing.db')

# Load ALL data from 2020 onwards (need history for calculations)
query = '''
SELECT 
    ge.GreyhoundID, 
    g.GreyhoundName,
    ge.RaceID, 
    ge.FirstSplitPosition, 
    ge.Box,
    ge.Position, 
    ge.StartingPrice, 
    ge.CareerPrizeMoney,
    ge.FinishTimeBenchmarkLengths,
    r.RaceNumber,
    r.Distance,
    rm.MeetingDate, 
    rm.MeetingAvgBenchmarkLengths,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
LEFT JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position IS NOT NULL
  AND ge.StartingPrice IS NOT NULL
  AND ge.FirstSplitPosition IS NOT NULL
  AND ge.FirstSplitPosition != ''
  AND rm.MeetingDate >= '2020-01-01'
ORDER BY ge.GreyhoundID, rm.MeetingDate, r.RaceID
'''

print('Loading data...')
df = pd.read_sql_query(query, conn)
conn.close()
print(f'Loaded {len(df):,} rows')

# Convert types
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
df['Year'] = pd.to_datetime(df['MeetingDate']).dt.year

# Calculate pace
df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])

# Historical metrics (ONLY using past data - no leakage)
print('Calculating historical metrics...')
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
df['CumPace'] = df.groupby('GreyhoundID')['TotalPace'].cumsum().shift(1)
df['HistAvgPace'] = df['CumPace'] / df['CumCount']

# Box adjustment
box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

# Filter to dogs with history (5+ prior races)
df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

# Rankings within each race
print('Calculating race rankings...')
df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1
df['IsPaceTop3'] = df['PaceRank'] <= 3

print(f'After filtering: {len(df):,} rows')
print('')

# Filter to 2025 only
df_2025 = df[df['Year'] == 2025].copy()
print(f'2025 races: {len(df_2025):,} rows')

# Odds range filter
ODDS_MIN = 15
ODDS_MAX = 50
df_2025 = df_2025[(df_2025['StartingPrice'] >= ODDS_MIN) & (df_2025['StartingPrice'] <= ODDS_MAX)]
print(f'After odds filter ${ODDS_MIN}-${ODDS_MAX}: {len(df_2025):,} rows')

# Strategy filters
df_2025['HasMoney'] = df_2025['CareerPrizeMoney'] >= 30000

# Apply all 3 strategies and mark which ones apply
df_2025['Strategy1_PIR_PaceLeader_Money'] = df_2025['IsPIRLeader'] & df_2025['IsPaceLeader'] & df_2025['HasMoney']
df_2025['Strategy2_PIR_PaceTop3_Money'] = df_2025['IsPIRLeader'] & df_2025['IsPaceTop3'] & df_2025['HasMoney']
df_2025['Strategy3_PIR_Money'] = df_2025['IsPIRLeader'] & df_2025['HasMoney']

# Get bets for each strategy
strat1 = df_2025[df_2025['Strategy1_PIR_PaceLeader_Money']].copy()
strat2 = df_2025[df_2025['Strategy2_PIR_PaceTop3_Money']].copy()
strat3 = df_2025[df_2025['Strategy3_PIR_Money']].copy()

def calc_stats(subset, name):
    if len(subset) == 0:
        return
    bets = len(subset)
    wins = subset['Win'].sum()
    returns = (subset[subset['Win']]['StartingPrice']).sum()
    profit = returns - bets
    roi = profit / bets * 100
    print(f'{name}:')
    print(f'  Bets: {bets}')
    print(f'  Wins: {wins} ({wins/bets*100:.1f}%)')
    print(f'  Profit: {profit:+.1f} units')
    print(f'  ROI: {roi:+.1f}%')
    print('')
    return bets, wins, profit, roi

print('')
print('='*80)
print('2025 BACKTEST RESULTS')
print('='*80)
print('')

calc_stats(strat1, 'Strategy 1: PIR Leader + Pace Leader + $30k')
calc_stats(strat2, 'Strategy 2: PIR Leader + Pace Top 3 + $30k')
calc_stats(strat3, 'Strategy 3: PIR Leader + $30k (no pace)')

# Export detailed CSV with all bets
print('='*80)
print('EXPORTING DETAILED CSV')
print('='*80)

# Use Strategy 3 as base (includes all bets from strategies 1 and 2)
all_bets = strat3.copy()

# Add strategy flags
all_bets['Strategy'] = 'PIR + Money Only'
all_bets.loc[all_bets['Strategy2_PIR_PaceTop3_Money'], 'Strategy'] = 'PIR + Pace Top 3 + Money'
all_bets.loc[all_bets['Strategy1_PIR_PaceLeader_Money'], 'Strategy'] = 'PIR + Pace Leader + Money'

# Calculate profit for each bet
all_bets['Profit'] = all_bets.apply(lambda r: r['StartingPrice'] - 1 if r['Win'] else -1, axis=1)

# Select and rename columns for export
export_df = all_bets[[
    'MeetingDate', 'TrackName', 'RaceNumber', 'GreyhoundName', 'Box',
    'StartingPrice', 'Position', 'Win', 'Profit',
    'HistAvgSplit', 'PredSplit', 'PIRRank',
    'HistAvgPace', 'PaceRank',
    'CareerPrizeMoney', 'CumCount',
    'Strategy'
]].copy()

export_df.columns = [
    'Date', 'Track', 'Race', 'Greyhound', 'Box',
    'Odds', 'Position', 'Win', 'Profit',
    'HistAvgSplit', 'PredictedSplit', 'PIRRank',
    'HistAvgPace', 'PaceRank',
    'CareerMoney', 'PriorRaces',
    'Strategy'
]

# Sort by date and race
export_df = export_df.sort_values(['Date', 'Track', 'Race'])

# Format numbers
export_df['HistAvgSplit'] = export_df['HistAvgSplit'].round(2)
export_df['PredictedSplit'] = export_df['PredictedSplit'].round(2)
export_df['HistAvgPace'] = export_df['HistAvgPace'].round(2)
export_df['Odds'] = export_df['Odds'].round(2)
export_df['Profit'] = export_df['Profit'].round(2)
export_df['CareerMoney'] = export_df['CareerMoney'].astype(int)
export_df['PriorRaces'] = export_df['PriorRaces'].astype(int)

# Export
csv_path = 'backtest_2025_all_bets.csv'
export_df.to_csv(csv_path, index=False)
print(f'Exported {len(export_df)} bets to {csv_path}')

# Summary by month
print('')
print('='*80)
print('MONTHLY BREAKDOWN (Strategy 1: PIR + Pace Leader + $30k)')
print('='*80)
strat1['Month'] = pd.to_datetime(strat1['MeetingDate']).dt.to_period('M')
monthly = strat1.groupby('Month').agg({
    'Win': ['count', 'sum'],
    'StartingPrice': lambda x: (x[strat1.loc[x.index, 'Win']]).sum() if strat1.loc[x.index, 'Win'].any() else 0
}).reset_index()
monthly.columns = ['Month', 'Bets', 'Wins', 'Returns']
monthly['Profit'] = monthly['Returns'] - monthly['Bets']
monthly['ROI'] = monthly['Profit'] / monthly['Bets'] * 100
monthly['StrikeRate'] = monthly['Wins'] / monthly['Bets'] * 100
print(monthly.to_string(index=False))

print('')
print('='*80)
print('CUMULATIVE P/L (Strategy 1: PIR + Pace Leader + $30k)')
print('='*80)
strat1_sorted = strat1.sort_values('MeetingDate')
strat1_sorted['CumProfit'] = strat1_sorted['Profit'].cumsum()
print(f'Starting: $0.00')
print(f'Final: ${strat1_sorted["CumProfit"].iloc[-1]:.2f} after {len(strat1_sorted)} bets')
print(f'Max Drawdown: ${strat1_sorted["CumProfit"].min():.2f}')
print(f'Peak: ${strat1_sorted["CumProfit"].max():.2f}')

print('')
print('DONE!')
