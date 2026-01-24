"""
Test PIR + Pace strategies across different odds brackets for 2025
"""
import sqlite3
import pandas as pd
import numpy as np

print('Loading data...')
conn = sqlite3.connect('greyhound_racing.db')

query = '''
SELECT 
    ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
    ge.Position, ge.StartingPrice, ge.CareerPrizeMoney,
    ge.FinishTimeBenchmarkLengths,
    rm.MeetingDate, rm.MeetingAvgBenchmarkLengths
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
conn.close()

# Convert types
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
df['Year'] = pd.to_datetime(df['MeetingDate']).dt.year
df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])

# Historical metrics
print('Calculating historical metrics...')
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
df['CumPace'] = df.groupby('GreyhoundID')['TotalPace'].cumsum().shift(1)
df['HistAvgPace'] = df['CumPace'] / df['CumCount']

box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

print('Calculating race rankings...')
df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1
df['IsPaceTop3'] = df['PaceRank'] <= 3
df['HasMoney'] = df['CareerPrizeMoney'] >= 30000

# Filter to 2025
df = df[df['Year'] == 2025]
print(f'2025 data: {len(df):,} rows')

print('')
print('='*90)
print('2025 STRATEGY RESULTS BY ODDS BRACKET')
print('='*90)

odds_brackets = [
    (1.5, 3.0, '$1.50-$3'),
    (3.0, 5.0, '$3-$5'),
    (5.0, 10.0, '$5-$10'),
    (10.0, 15.0, '$10-$15'),
    (15.0, 30.0, '$15-$30'),
    (30.0, 50.0, '$30-$50'),
]

strategies = [
    ('PIR+Pace Leader+$30k', lambda d: d['IsPIRLeader'] & d['IsPaceLeader'] & d['HasMoney']),
    ('PIR+Pace Top3+$30k', lambda d: d['IsPIRLeader'] & d['IsPaceTop3'] & d['HasMoney']),
    ('PIR+$30k (no pace)', lambda d: d['IsPIRLeader'] & d['HasMoney']),
]

for strat_name, strat_func in strategies:
    print(f'\n{strat_name}')
    print('-'*90)
    print(f'{"Odds Range":<15} {"Bets":>8} {"Wins":>8} {"Strike%":>10} {"Profit":>10} {"ROI%":>10}')
    print('-'*90)
    
    for odds_min, odds_max, label in odds_brackets:
        subset = df[(df['StartingPrice'] >= odds_min) & (df['StartingPrice'] < odds_max)]
        mask = strat_func(subset)
        bets_df = subset[mask]
        
        bets = len(bets_df)
        if bets == 0:
            print(f'{label:<15} {0:>8} {0:>8} {"N/A":>10} {0:>+10.1f} {"N/A":>10}')
            continue
            
        wins = bets_df['Win'].sum()
        returns = bets_df[bets_df['Win']]['StartingPrice'].sum()
        profit = returns - bets
        roi = profit / bets * 100
        strike = wins / bets * 100
        
        print(f'{label:<15} {bets:>8} {wins:>8} {strike:>10.1f} {profit:>+10.1f} {roi:>+10.1f}')
    
    # Total across all brackets
    all_bets = df[strat_func(df)]
    total_bets = len(all_bets)
    total_wins = all_bets['Win'].sum()
    total_returns = all_bets[all_bets['Win']]['StartingPrice'].sum()
    total_profit = total_returns - total_bets
    total_roi = total_profit / total_bets * 100 if total_bets > 0 else 0
    total_strike = total_wins / total_bets * 100 if total_bets > 0 else 0
    print('-'*90)
    print(f'{"ALL ODDS":<15} {total_bets:>8} {total_wins:>8} {total_strike:>10.1f} {total_profit:>+10.1f} {total_roi:>+10.1f}')

print()
print('='*90)
print('SUMMARY: Best performing odds brackets')
print('='*90)
