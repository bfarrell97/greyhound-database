"""
Walk-Forward Validation of Inverse-Odds Staking Strategy
Tests:
1. Year-by-year out-of-sample performance
2. Monte Carlo shuffle test
3. Drawdown analysis
4. Comparison with flat staking
"""
import sqlite3
import pandas as pd
import numpy as np

print('='*90)
print('WALK-FORWARD VALIDATION: INVERSE-ODDS STAKING')
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

df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1
df['IsPaceTop3'] = df['PaceRank'] <= 3
df['HasMoney'] = df['CareerPrizeMoney'] >= 30000

# Filter to odds range
df = df[(df['StartingPrice'] >= 1.50) & (df['StartingPrice'] <= 30)]

print(f'Filtered data: {len(df):,} rows')
print()

# Staking functions
def get_inverse_multiplier(odds):
    if odds < 3:
        return 0.5
    elif odds < 5:
        return 0.75
    elif odds < 10:
        return 1.0
    elif odds < 20:
        return 1.5
    else:
        return 2.0

def calc_results(bets_df, use_inverse=False):
    """Calculate P/L with flat or inverse-odds staking"""
    if len(bets_df) == 0:
        return 0, 0, 0, 0, 0
    
    bets_df = bets_df.sort_values('MeetingDate').copy()
    
    if use_inverse:
        bets_df['Stake'] = bets_df['StartingPrice'].apply(get_inverse_multiplier)
    else:
        bets_df['Stake'] = 1.0
    
    bets_df['PL'] = bets_df.apply(
        lambda r: r['Stake'] * (r['StartingPrice'] - 1) if r['Win'] else -r['Stake'], 
        axis=1
    )
    
    total_staked = bets_df['Stake'].sum()
    profit = bets_df['PL'].sum()
    roi = profit / total_staked * 100 if total_staked > 0 else 0
    wins = bets_df['Win'].sum()
    
    # Max drawdown
    bets_df['CumPL'] = bets_df['PL'].cumsum()
    bets_df['Peak'] = bets_df['CumPL'].cummax()
    bets_df['DD'] = bets_df['CumPL'] - bets_df['Peak']
    max_dd = bets_df['DD'].min()
    
    return len(bets_df), wins, profit, roi, max_dd

# Define strategies
strategies = {
    'PIR + Pace Leader + $30k': lambda d: d['IsPIRLeader'] & d['IsPaceLeader'] & d['HasMoney'],
    'PIR + Pace Top 3 + $30k': lambda d: d['IsPIRLeader'] & d['IsPaceTop3'] & d['HasMoney'],
}

# ============================================================
# TEST 1: YEAR-BY-YEAR WALK-FORWARD VALIDATION
# ============================================================
print('='*90)
print('TEST 1: YEAR-BY-YEAR WALK-FORWARD VALIDATION')
print('='*90)
print()

for strat_name, strat_func in strategies.items():
    print(f'{strat_name}')
    print('-'*90)
    print(f'{"Year":<8} {"Bets":>8} {"Wins":>8} {"Flat Profit":>12} {"Flat ROI%":>10} {"Inv Profit":>12} {"Inv ROI%":>10}')
    print('-'*90)
    
    flat_total_profit = 0
    flat_total_staked = 0
    inv_total_profit = 0
    inv_total_staked = 0
    yearly_flat_roi = []
    yearly_inv_roi = []
    
    for year in [2021, 2022, 2023, 2024, 2025]:
        year_data = df[df['Year'] == year]
        mask = strat_func(year_data)
        bets = year_data[mask]
        
        if len(bets) < 5:
            continue
        
        # Flat staking
        n_bets, wins, flat_profit, flat_roi, _ = calc_results(bets, use_inverse=False)
        flat_total_profit += flat_profit
        flat_total_staked += n_bets
        yearly_flat_roi.append(flat_roi)
        
        # Inverse-odds staking
        _, _, inv_profit, inv_roi, _ = calc_results(bets, use_inverse=True)
        inv_total_staked += bets['StartingPrice'].apply(get_inverse_multiplier).sum()
        inv_total_profit += inv_profit
        yearly_inv_roi.append(inv_roi)
        
        print(f'{year:<8} {n_bets:>8} {wins:>8} {flat_profit:>+12.1f} {flat_roi:>+10.1f} {inv_profit:>+12.1f} {inv_roi:>+10.1f}')
    
    flat_overall_roi = flat_total_profit / flat_total_staked * 100 if flat_total_staked > 0 else 0
    inv_overall_roi = inv_total_profit / inv_total_staked * 100 if inv_total_staked > 0 else 0
    
    print('-'*90)
    print(f'{"TOTAL":<8} {"":>8} {"":>8} {flat_total_profit:>+12.1f} {flat_overall_roi:>+10.1f} {inv_total_profit:>+12.1f} {inv_overall_roi:>+10.1f}')
    
    flat_profitable = sum(1 for r in yearly_flat_roi if r > 0)
    inv_profitable = sum(1 for r in yearly_inv_roi if r > 0)
    print(f'\nFlat: {flat_profitable}/{len(yearly_flat_roi)} years profitable')
    print(f'Inverse-Odds: {inv_profitable}/{len(yearly_inv_roi)} years profitable')
    print()

# ============================================================
# TEST 2: MONTE CARLO SHUFFLE TEST
# ============================================================
print('='*90)
print('TEST 2: MONTE CARLO SHUFFLE TEST (Inverse-Odds Staking)')
print('='*90)
print('Testing if inverse-odds staking edge is statistically significant')
print()

N_SIMULATIONS = 1000

# Get overall win rate by odds bracket for simulation
overall_win_rates = {}
for low, high in [(1.5, 3), (3, 5), (5, 10), (10, 20), (20, 30)]:
    bracket = df[(df['StartingPrice'] >= low) & (df['StartingPrice'] < high)]
    if len(bracket) > 0:
        overall_win_rates[(low, high)] = bracket['Win'].mean()

def get_win_rate_for_odds(odds):
    for (low, high), wr in overall_win_rates.items():
        if low <= odds < high:
            return wr
    return 0.15  # default

for strat_name, strat_func in strategies.items():
    print(f'{strat_name}')
    print('-'*90)
    
    # Get all bets (2021-2025)
    test_data = df[df['Year'] >= 2021]
    mask = strat_func(test_data)
    actual_bets = test_data[mask].copy()
    
    if len(actual_bets) == 0:
        print('  No bets')
        continue
    
    # Actual results with inverse-odds
    _, _, actual_profit, actual_roi, _ = calc_results(actual_bets, use_inverse=True)
    actual_staked = actual_bets['StartingPrice'].apply(get_inverse_multiplier).sum()
    
    print(f'  Actual: {len(actual_bets)} bets, Staked: {actual_staked:.1f}, Profit: {actual_profit:+.1f}, ROI: {actual_roi:+.1f}%')
    
    # Monte Carlo: randomize wins based on overall market win rates
    odds_array = actual_bets['StartingPrice'].values
    stakes_array = np.array([get_inverse_multiplier(o) for o in odds_array])
    win_probs = np.array([get_win_rate_for_odds(o) for o in odds_array])
    
    np.random.seed(42)
    
    # Vectorized simulation
    random_wins = np.random.random((N_SIMULATIONS, len(actual_bets))) < win_probs
    
    # Calculate P/L for each simulation
    # If win: stake * (odds - 1), if lose: -stake
    win_returns = stakes_array * (odds_array - 1)  # Returns if win
    loss_returns = -stakes_array  # Returns if lose
    
    # For each simulation, calculate total P/L
    random_pls = np.where(random_wins, win_returns, loss_returns).sum(axis=1)
    random_rois = random_pls / actual_staked * 100
    
    # Statistics
    mean_random_roi = np.mean(random_rois)
    std_random_roi = np.std(random_rois)
    z_score = (actual_roi - mean_random_roi) / std_random_roi if std_random_roi > 0 else 0
    better_count = np.sum(random_rois >= actual_roi)
    p_value = better_count / N_SIMULATIONS
    
    print(f'  Random Mean ROI: {mean_random_roi:+.1f}% (std: {std_random_roi:.1f}%)')
    print(f'  Z-Score: {z_score:.2f}')
    print(f'  P-Value: {p_value:.6f} ({better_count}/{N_SIMULATIONS} random trials beat actual)')
    
    if p_value < 0.01:
        print(f'  >>> HIGHLY SIGNIFICANT (p < 0.01) - Edge is REAL <<<')
    elif p_value < 0.05:
        print(f'  >>> SIGNIFICANT (p < 0.05) - Edge is likely real <<<')
    else:
        print(f'  >>> NOT SIGNIFICANT - Edge may be due to chance <<<')
    print()

# ============================================================
# TEST 3: DRAWDOWN COMPARISON
# ============================================================
print('='*90)
print('TEST 3: DRAWDOWN ANALYSIS (Flat vs Inverse-Odds)')
print('='*90)
print()

for strat_name, strat_func in strategies.items():
    print(f'{strat_name}')
    print('-'*60)
    
    test_data = df[df['Year'] >= 2021]
    mask = strat_func(test_data)
    bets = test_data[mask].copy()
    
    if len(bets) == 0:
        continue
    
    # Flat staking analysis
    _, _, flat_profit, flat_roi, flat_dd = calc_results(bets, use_inverse=False)
    
    # Inverse-odds staking analysis
    _, _, inv_profit, inv_roi, inv_dd = calc_results(bets, use_inverse=True)
    
    print(f'{"Metric":<25} {"Flat Stake":>15} {"Inverse-Odds":>15}')
    print('-'*60)
    print(f'{"Final Profit":<25} {flat_profit:>+15.1f} {inv_profit:>+15.1f}')
    print(f'{"ROI %":<25} {flat_roi:>+15.1f} {inv_roi:>+15.1f}')
    print(f'{"Max Drawdown":<25} {flat_dd:>15.1f} {inv_dd:>15.1f}')
    print(f'{"Recovery Ratio":<25} {flat_profit/abs(flat_dd):>15.1f}x {inv_profit/abs(inv_dd):>15.1f}x')
    print()

# ============================================================
# TEST 4: CONSISTENCY BY ODDS BRACKET
# ============================================================
print('='*90)
print('TEST 4: INVERSE-ODDS CONSISTENCY BY BRACKET (2021-2025)')
print('='*90)
print()

brackets = [(1.5, 3, '$1.50-$3'), (3, 5, '$3-$5'), (5, 10, '$5-$10'), 
            (10, 20, '$10-$20'), (20, 30, '$20-$30')]

for strat_name, strat_func in strategies.items():
    print(f'{strat_name}')
    print('-'*90)
    print(f'{"Odds":<12} {"Bets":>8} {"Wins":>8} {"Stake Mult":>12} {"Profit":>12} {"ROI%":>10} {"Status":>12}')
    print('-'*90)
    
    test_data = df[df['Year'] >= 2021]
    mask = strat_func(test_data)
    all_bets = test_data[mask]
    
    profitable_brackets = 0
    
    for low, high, label in brackets:
        bracket_bets = all_bets[(all_bets['StartingPrice'] >= low) & (all_bets['StartingPrice'] < high)]
        
        if len(bracket_bets) < 10:
            print(f'{label:<12} {len(bracket_bets):>8} {"":>8} {"":>12} {"":>12} {"N/A":>10} {"LOW VOLUME":>12}')
            continue
        
        mult = get_inverse_multiplier((low + high) / 2)
        _, wins, profit, roi, _ = calc_results(bracket_bets, use_inverse=True)
        
        status = "PROFIT" if roi > 0 else "LOSS"
        if roi > 0:
            profitable_brackets += 1
        
        print(f'{label:<12} {len(bracket_bets):>8} {wins:>8} {mult:>12.2f}x {profit:>+12.1f} {roi:>+10.1f} {status:>12}')
    
    print('-'*90)
    print(f'Profitable brackets: {profitable_brackets}/{len(brackets)}')
    print()

# ============================================================
# FINAL VERDICT
# ============================================================
print('='*90)
print('VALIDATION VERDICT: INVERSE-ODDS STAKING')
print('='*90)
print()
print('Criteria for validation:')
print('  1. Walk-Forward: Profitable in majority of years (out-of-sample)')
print('  2. Monte Carlo: P-value < 0.05 (edge not due to chance)')
print('  3. Drawdown: Recovery ratio > 1.5x')
print('  4. Consistency: Profitable in majority of odds brackets')
print()
print('If all criteria pass, inverse-odds staking is validated.')
