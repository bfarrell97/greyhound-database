"""
Staking Strategy Analysis for PIR + Pace Models
Compare: Flat stakes, Kelly Criterion, Proportional, Level stakes by odds bracket
"""
import sqlite3
import pandas as pd
import numpy as np

print('='*90)
print('STAKING STRATEGY ANALYSIS - PIR + PACE MODELS')
print('='*90)
print()

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

print('Loading data...')
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

df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1
df['IsPaceTop3'] = df['PaceRank'] <= 3
df['HasMoney'] = df['CareerPrizeMoney'] >= 30000

# Filter to test period (2021-2025) and odds range
df = df[(df['Year'] >= 2021) & 
        (df['StartingPrice'] >= 1.50) & 
        (df['StartingPrice'] <= 30)]

print(f'Test data: {len(df):,} rows (2021-2025, $1.50-$30)')
print()

# Get bets for both strategies
strat1_mask = df['IsPIRLeader'] & df['IsPaceLeader'] & df['HasMoney']
strat2_mask = df['IsPIRLeader'] & df['IsPaceTop3'] & df['HasMoney']

strategies = {
    'PIR + Pace Leader + $30k': df[strat1_mask].copy(),
    'PIR + Pace Top 3 + $30k': df[strat2_mask].copy()
}

# Define staking methods
def flat_stake(bets_df, stake=1.0):
    """Flat $1 per bet"""
    bets_df = bets_df.sort_values('MeetingDate').copy()
    bets_df['Stake'] = stake
    bets_df['PL'] = bets_df.apply(lambda r: r['Stake'] * (r['StartingPrice'] - 1) if r['Win'] else -r['Stake'], axis=1)
    return bets_df['PL'].sum(), bets_df['Stake'].sum(), bets_df

def kelly_stake(bets_df, bankroll=100, fraction=0.25):
    """
    Fractional Kelly Criterion
    Kelly % = (bp - q) / b where b=odds-1, p=win prob, q=1-p
    Use historical win rate by odds bracket as probability estimate
    """
    bets_df = bets_df.sort_values('MeetingDate').copy()
    
    # Calculate win rates by odds bracket for probability estimates
    odds_brackets = [(1.5, 3), (3, 5), (5, 10), (10, 20), (20, 30)]
    win_rates = {}
    for low, high in odds_brackets:
        bracket_bets = bets_df[(bets_df['StartingPrice'] >= low) & (bets_df['StartingPrice'] < high)]
        if len(bracket_bets) > 0:
            win_rates[(low, high)] = bracket_bets['Win'].mean()
    
    stakes = []
    pls = []
    running_bank = bankroll
    
    for _, row in bets_df.iterrows():
        odds = row['StartingPrice']
        
        # Find win rate for this odds bracket
        p = 0.3  # default
        for (low, high), wr in win_rates.items():
            if low <= odds < high:
                p = wr
                break
        
        b = odds - 1
        q = 1 - p
        
        # Kelly formula
        kelly_pct = (b * p - q) / b if b > 0 else 0
        kelly_pct = max(0, kelly_pct)  # No negative stakes
        
        # Fractional Kelly (safer)
        stake_pct = kelly_pct * fraction
        stake = running_bank * stake_pct
        stake = max(0.1, min(stake, running_bank * 0.1))  # Cap at 10% of bank
        
        if row['Win']:
            pl = stake * (odds - 1)
        else:
            pl = -stake
        
        running_bank += pl
        stakes.append(stake)
        pls.append(pl)
    
    bets_df['Stake'] = stakes
    bets_df['PL'] = pls
    return sum(pls), sum(stakes), bets_df

def proportional_stake(bets_df, target_win=10):
    """
    Stake to win fixed amount (e.g., $10)
    Stake = target_win / (odds - 1)
    """
    bets_df = bets_df.sort_values('MeetingDate').copy()
    bets_df['Stake'] = target_win / (bets_df['StartingPrice'] - 1)
    bets_df['PL'] = bets_df.apply(lambda r: r['Stake'] * (r['StartingPrice'] - 1) if r['Win'] else -r['Stake'], axis=1)
    return bets_df['PL'].sum(), bets_df['Stake'].sum(), bets_df

def odds_weighted_stake(bets_df, base_stake=1.0):
    """
    Higher stakes on shorter odds (more likely to win)
    Stake multiplier based on odds bracket
    """
    bets_df = bets_df.sort_values('MeetingDate').copy()
    
    def get_multiplier(odds):
        if odds < 3:
            return 2.0  # Double stake on short odds
        elif odds < 5:
            return 1.5
        elif odds < 10:
            return 1.0
        elif odds < 20:
            return 0.75
        else:
            return 0.5  # Half stake on long odds
    
    bets_df['Stake'] = bets_df['StartingPrice'].apply(get_multiplier) * base_stake
    bets_df['PL'] = bets_df.apply(lambda r: r['Stake'] * (r['StartingPrice'] - 1) if r['Win'] else -r['Stake'], axis=1)
    return bets_df['PL'].sum(), bets_df['Stake'].sum(), bets_df

def inverse_odds_stake(bets_df, base_stake=1.0):
    """
    Higher stakes on LONGER odds (better value in our model)
    Based on our analysis showing higher ROI at longer odds
    """
    bets_df = bets_df.sort_values('MeetingDate').copy()
    
    def get_multiplier(odds):
        if odds < 3:
            return 0.5  # Half stake on short odds
        elif odds < 5:
            return 0.75
        elif odds < 10:
            return 1.0
        elif odds < 20:
            return 1.5
        else:
            return 2.0  # Double stake on long odds
    
    bets_df['Stake'] = bets_df['StartingPrice'].apply(get_multiplier) * base_stake
    bets_df['PL'] = bets_df.apply(lambda r: r['Stake'] * (r['StartingPrice'] - 1) if r['Win'] else -r['Stake'], axis=1)
    return bets_df['PL'].sum(), bets_df['Stake'].sum(), bets_df

# Compare staking strategies
print('='*90)
print('STAKING STRATEGY COMPARISON (2021-2025)')
print('='*90)

staking_methods = [
    ('Flat $1', flat_stake),
    ('Kelly (25%)', kelly_stake),
    ('Proportional (Win $10)', proportional_stake),
    ('Odds-Weighted (Higher on shorts)', odds_weighted_stake),
    ('Inverse-Odds (Higher on longs)', inverse_odds_stake),
]

for strat_name, bets_df in strategies.items():
    print(f'\n{strat_name}')
    print(f'Total bets: {len(bets_df)}, Wins: {bets_df["Win"].sum()} ({bets_df["Win"].mean()*100:.1f}%)')
    print('-'*90)
    print(f'{"Staking Method":<35} {"Total Staked":>15} {"Profit":>15} {"ROI%":>10} {"Max DD":>10}')
    print('-'*90)
    
    results = []
    
    for method_name, method_func in staking_methods:
        profit, staked, result_df = method_func(bets_df.copy())
        roi = profit / staked * 100 if staked > 0 else 0
        
        # Calculate max drawdown
        result_df['CumPL'] = result_df['PL'].cumsum()
        result_df['Peak'] = result_df['CumPL'].cummax()
        result_df['DD'] = result_df['CumPL'] - result_df['Peak']
        max_dd = result_df['DD'].min()
        
        print(f'{method_name:<35} {staked:>15.1f} {profit:>+15.1f} {roi:>+10.1f} {max_dd:>10.1f}')
        results.append((method_name, roi, profit, max_dd))
    
    # Find best method
    best = max(results, key=lambda x: x[1])
    print('-'*90)
    print(f'BEST: {best[0]} with {best[1]:+.1f}% ROI')

# Detailed analysis of best performer
print()
print('='*90)
print('DETAILED ANALYSIS: FLAT STAKE vs INVERSE-ODDS')
print('='*90)
print()

for strat_name, bets_df in strategies.items():
    print(f'{strat_name}')
    print('-'*60)
    
    # Compare by odds bracket
    print(f'{"Odds":<12} {"Flat ROI%":>12} {"Inv-Odds ROI%":>15} {"Flat Profit":>12} {"Inv Profit":>12}')
    print('-'*60)
    
    brackets = [(1.5, 3, '$1.50-$3'), (3, 5, '$3-$5'), (5, 10, '$5-$10'), 
                (10, 20, '$10-$20'), (20, 30, '$20-$30')]
    
    for low, high, label in brackets:
        bracket_df = bets_df[(bets_df['StartingPrice'] >= low) & (bets_df['StartingPrice'] < high)]
        if len(bracket_df) < 5:
            continue
        
        # Flat
        flat_profit, flat_staked, _ = flat_stake(bracket_df.copy())
        flat_roi = flat_profit / flat_staked * 100 if flat_staked > 0 else 0
        
        # Inverse odds
        inv_profit, inv_staked, _ = inverse_odds_stake(bracket_df.copy())
        inv_roi = inv_profit / inv_staked * 100 if inv_staked > 0 else 0
        
        print(f'{label:<12} {flat_roi:>+12.1f} {inv_roi:>+15.1f} {flat_profit:>+12.1f} {inv_profit:>+12.1f}')
    
    print()

print('='*90)
print('RECOMMENDATION')
print('='*90)
print()
print('Based on the analysis:')
print()
print('1. FLAT STAKE is the simplest and performs well')
print('2. INVERSE-ODDS (higher stakes on longer odds) can boost profits')
print('   because our model has higher ROI at longer odds')
print('3. KELLY is mathematically optimal but requires accurate probability estimates')
print()
print('For practical use: Start with FLAT STAKE for simplicity and consistency.')
print('Advanced: Consider INVERSE-ODDS weighting to capitalize on higher ROI at long odds.')
