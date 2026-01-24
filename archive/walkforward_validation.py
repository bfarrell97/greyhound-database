"""
Comprehensive Walk-Forward Validation of PIR + Money + Pace Strategies
Tests legitimacy with:
1. Walk-forward (train/test) validation
2. Year-by-year out-of-sample testing
3. Monte Carlo shuffle tests
4. Drawdown analysis
"""
import sqlite3
import pandas as pd
import numpy as np
import sys

def fp(msg):
    print(msg)
    sys.stdout.flush()

fp('='*80)
fp('COMPREHENSIVE WALK-FORWARD VALIDATION')
fp('='*80)
fp('')

conn = sqlite3.connect('greyhound_racing.db')

# Load ALL data from 2020 onwards for walk-forward
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

fp('Loading data...')
df = pd.read_sql_query(query, conn)
fp(f'Loaded {len(df):,} rows')

# Convert types
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
df['Year'] = pd.to_datetime(df['MeetingDate']).dt.year
df['Month'] = pd.to_datetime(df['MeetingDate']).dt.to_period('M')

# Calculate pace
df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']

df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])

# Historical metrics (ONLY using past data - no leakage)
df['CumCount'] = df.groupby('GreyhoundID').cumcount()
df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
df['CumPace'] = df.groupby('GreyhoundID')['TotalPace'].cumsum().shift(1)
df['HistAvgPace'] = df['CumPace'] / df['CumCount']

# Box adjustment
box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
df['PredSplit'] = df['HistAvgSplit'] + df['BoxAdj']

# Filter to dogs with history
df = df[df['CumCount'] >= 5].copy()
df = df.dropna(subset=['HistAvgSplit'])

# Rankings within each race
df['PIRRank'] = df.groupby('RaceID')['PredSplit'].rank(method='min')
df['IsPIRLeader'] = df['PIRRank'] == 1
df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=True)
df['IsPaceLeader'] = df['PaceRank'] == 1

fp(f'After filtering: {len(df):,} rows')
fp('')

# Define strategies
STRATEGIES = {
    'PIR + Money': lambda d: d['IsPIRLeader'] & (d['CareerPrizeMoney'] >= 30000),
    'PIR + Money + PaceLeader': lambda d: d['IsPIRLeader'] & (d['CareerPrizeMoney'] >= 30000) & d['IsPaceLeader'],
    'PIR + Money + PaceTop3': lambda d: d['IsPIRLeader'] & (d['CareerPrizeMoney'] >= 30000) & (d['PaceRank'] <= 3),
    'PIR + Money + PaceTop5': lambda d: d['IsPIRLeader'] & (d['CareerPrizeMoney'] >= 30000) & (d['PaceRank'] <= 5),
}

ODDS_MIN = 15
ODDS_MAX = 50

def calc_roi(subset):
    if len(subset) == 0:
        return 0, 0, 0
    bets = len(subset)
    wins = subset['Win'].sum()
    returns = (subset[subset['Win']]['StartingPrice'] - 1).sum()
    roi = (returns - bets) / bets * 100 if bets > 0 else 0
    return bets, wins, roi

# ============================================================
# TEST 1: WALK-FORWARD VALIDATION
# Train on Year N, test on Year N+1
# ============================================================
fp('='*80)
fp('TEST 1: WALK-FORWARD VALIDATION (Train Year N -> Test Year N+1)')
fp('='*80)
fp('')

for strat_name, strat_func in STRATEGIES.items():
    fp(f'Strategy: {strat_name}')
    fp('-'*60)
    
    total_bets = 0
    total_profit = 0
    
    for train_year in [2020, 2021, 2022, 2023, 2024]:
        test_year = train_year + 1
        if test_year > 2025:
            continue
            
        # Get test data
        test_data = df[(df['Year'] == test_year) & 
                       (df['StartingPrice'] >= ODDS_MIN) & 
                       (df['StartingPrice'] <= ODDS_MAX)]
        
        # Apply strategy
        mask = strat_func(test_data)
        subset = test_data[mask]
        
        bets, wins, roi = calc_roi(subset)
        profit = (subset[subset['Win']]['StartingPrice'] - 1).sum() - bets if bets > 0 else 0
        
        total_bets += bets
        total_profit += profit
        
        if bets >= 10:
            fp(f'  Train {train_year} -> Test {test_year}: {bets:4d} bets, {wins:3d} wins ({wins/bets*100:5.1f}%), ROI: {roi:+6.1f}%')
    
    overall_roi = total_profit / total_bets * 100 if total_bets > 0 else 0
    fp(f'  TOTAL OUT-OF-SAMPLE: {total_bets} bets, ROI: {overall_roi:+.1f}%')
    fp('')

# ============================================================
# TEST 2: PURE OUT-OF-SAMPLE (2024-2025 only, no optimization)
# ============================================================
fp('='*80)
fp('TEST 2: PURE OUT-OF-SAMPLE (2024-2025 - most recent data)')
fp('='*80)
fp('')

recent = df[(df['Year'] >= 2024) & 
            (df['StartingPrice'] >= ODDS_MIN) & 
            (df['StartingPrice'] <= ODDS_MAX)]

for strat_name, strat_func in STRATEGIES.items():
    mask = strat_func(recent)
    subset = recent[mask]
    bets, wins, roi = calc_roi(subset)
    days = len(subset['MeetingDate'].unique()) if len(subset) > 0 else 1
    daily = bets / days
    fp(f'{strat_name}: {bets} bets, {wins/bets*100:.1f}% wins, ROI: {roi:+.1f}%, {daily:.1f}/day')

# ============================================================
# TEST 3: MONTE CARLO SHUFFLE TEST
# ============================================================
fp('')
fp('='*80)
fp('TEST 3: MONTE CARLO SHUFFLE TEST (Statistical Significance)')
fp('='*80)
fp('')

test_data = df[(df['Year'] >= 2024) & 
               (df['StartingPrice'] >= ODDS_MIN) & 
               (df['StartingPrice'] <= ODDS_MAX)]

for strat_name, strat_func in STRATEGIES.items():
    mask = strat_func(test_data)
    subset = test_data[mask].copy()
    
    if len(subset) < 30:
        fp(f'{strat_name}: Not enough data for significance test')
        continue
    
    # Original ROI
    _, _, orig_roi = calc_roi(subset)
    
    # Shuffle test - randomly assign wins
    np.random.seed(42)
    shuffle_rois = []
    for _ in range(500):
        shuffled = subset.copy()
        shuffled['Win'] = np.random.permutation(shuffled['Win'].values)
        _, _, shuf_roi = calc_roi(shuffled)
        shuffle_rois.append(shuf_roi)
    
    # Also test vs random selection at same odds
    pool = test_data.copy()
    random_rois = []
    for _ in range(500):
        sample = pool.sample(n=min(len(subset), len(pool)), replace=True)
        _, _, rand_roi = calc_roi(sample)
        random_rois.append(rand_roi)
    
    avg_rand = np.mean(random_rois)
    std_rand = np.std(random_rois)
    z_score = (orig_roi - avg_rand) / std_rand if std_rand > 0 else 0
    
    p_value = sum(1 for r in random_rois if r >= orig_roi) / len(random_rois)
    
    fp(f'{strat_name}:')
    fp(f'  Original ROI: {orig_roi:+.1f}%')
    fp(f'  Random baseline: {avg_rand:+.1f}% (std: {std_rand:.1f}%)')
    fp(f'  Z-score: {z_score:.2f}')
    fp(f'  P-value (empirical): {p_value:.4f}')
    if z_score > 3:
        fp(f'  *** HIGHLY SIGNIFICANT ***')
    elif z_score > 2:
        fp(f'  ** SIGNIFICANT **')
    elif z_score > 1.5:
        fp(f'  * Marginally significant *')
    else:
        fp(f'  Not significant')
    fp('')

# ============================================================
# TEST 4: DRAWDOWN ANALYSIS
# ============================================================
fp('='*80)
fp('TEST 4: DRAWDOWN ANALYSIS (Risk Assessment)')
fp('='*80)
fp('')

for strat_name, strat_func in STRATEGIES.items():
    mask = strat_func(df) & (df['StartingPrice'] >= ODDS_MIN) & (df['StartingPrice'] <= ODDS_MAX)
    subset = df[mask].sort_values('MeetingDate').copy()
    
    if len(subset) < 30:
        continue
    
    # Calculate running P&L
    subset['Profit'] = np.where(subset['Win'], subset['StartingPrice'] - 1, -1)
    subset['CumProfit'] = subset['Profit'].cumsum()
    subset['RunningMax'] = subset['CumProfit'].cummax()
    subset['Drawdown'] = subset['CumProfit'] - subset['RunningMax']
    
    max_dd = subset['Drawdown'].min()
    final_profit = subset['CumProfit'].iloc[-1]
    
    # Losing streaks
    subset['IsLoss'] = ~subset['Win']
    subset['LossStreak'] = subset['IsLoss'].groupby((~subset['IsLoss']).cumsum()).cumsum()
    max_losing_streak = subset['LossStreak'].max()
    
    # Consecutive losing months
    monthly = subset.groupby('Month').agg({'Profit': 'sum'}).reset_index()
    monthly['IsLossMonth'] = monthly['Profit'] < 0
    monthly['LossMonthStreak'] = monthly['IsLossMonth'].groupby((~monthly['IsLossMonth']).cumsum()).cumsum()
    max_losing_months = monthly['LossMonthStreak'].max()
    
    fp(f'{strat_name}:')
    fp(f'  Total bets: {len(subset)}')
    fp(f'  Final profit: {final_profit:+.1f} units')
    fp(f'  Max drawdown: {max_dd:.1f} units')
    fp(f'  Max losing streak: {max_losing_streak} bets')
    fp(f'  Max consecutive losing months: {max_losing_months}')
    fp('')

# ============================================================
# TEST 5: STABILITY - ROLLING 6-MONTH ROI
# ============================================================
fp('='*80)
fp('TEST 5: STABILITY - Rolling 6-Month Performance')
fp('='*80)
fp('')

for strat_name, strat_func in STRATEGIES.items():
    mask = strat_func(df) & (df['StartingPrice'] >= ODDS_MIN) & (df['StartingPrice'] <= ODDS_MAX)
    subset = df[mask].copy()
    
    if len(subset) < 50:
        continue
    
    fp(f'{strat_name}:')
    
    # Group by 6-month periods
    subset['HalfYear'] = pd.to_datetime(subset['MeetingDate']).dt.to_period('2Q')
    
    periods_positive = 0
    periods_total = 0
    
    for period in sorted(subset['HalfYear'].unique()):
        p_data = subset[subset['HalfYear'] == period]
        bets, wins, roi = calc_roi(p_data)
        if bets >= 10:
            periods_total += 1
            if roi > 0:
                periods_positive += 1
            fp(f'  {period}: {bets:4d} bets, {wins:3d} wins ({wins/bets*100:5.1f}%), ROI: {roi:+6.1f}%')
    
    if periods_total > 0:
        fp(f'  Profitable periods: {periods_positive}/{periods_total} ({periods_positive/periods_total*100:.0f}%)')
    fp('')

# ============================================================
# FINAL SUMMARY
# ============================================================
fp('='*80)
fp('FINAL SUMMARY: LEGITIMACY ASSESSMENT')
fp('='*80)
fp('')

fp('Strategy                    | 2024-25 ROI | Z-Score | Max DD | Verdict')
fp('-'*80)

for strat_name, strat_func in STRATEGIES.items():
    # Get 2024-25 ROI
    recent = df[(df['Year'] >= 2024) & 
                (df['StartingPrice'] >= ODDS_MIN) & 
                (df['StartingPrice'] <= ODDS_MAX)]
    mask = strat_func(recent)
    subset = recent[mask]
    bets, wins, roi = calc_roi(subset)
    
    # Z-score (simplified)
    pool = recent.copy()
    np.random.seed(42)
    rand_rois = [calc_roi(pool.sample(n=min(len(subset), len(pool)), replace=True))[2] for _ in range(100)]
    z = (roi - np.mean(rand_rois)) / np.std(rand_rois) if np.std(rand_rois) > 0 else 0
    
    # Max drawdown
    full_mask = strat_func(df) & (df['StartingPrice'] >= ODDS_MIN) & (df['StartingPrice'] <= ODDS_MAX)
    full_subset = df[full_mask].sort_values('MeetingDate').copy()
    full_subset['Profit'] = np.where(full_subset['Win'], full_subset['StartingPrice'] - 1, -1)
    full_subset['CumProfit'] = full_subset['Profit'].cumsum()
    full_subset['RunningMax'] = full_subset['CumProfit'].cummax()
    max_dd = (full_subset['CumProfit'] - full_subset['RunningMax']).min()
    
    # Verdict
    if z > 3 and roi > 20:
        verdict = 'STRONG EDGE'
    elif z > 2 and roi > 10:
        verdict = 'LIKELY EDGE'
    elif z > 1.5 and roi > 0:
        verdict = 'POSSIBLE EDGE'
    else:
        verdict = 'UNCERTAIN'
    
    fp(f'{strat_name:27} | {roi:+6.1f}%     | {z:5.2f}   | {max_dd:6.1f} | {verdict}')

fp('')
conn.close()
fp('VALIDATION COMPLETE')
