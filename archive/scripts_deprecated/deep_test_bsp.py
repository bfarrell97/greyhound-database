"""
Deep Testing: Flat Staking Strategy at BSP
- Walk-Forward Testing (Rolling Windows)
- K-Fold Cross Validation
- Monte Carlo Simulation
- Drawdown Analysis
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import random

DB_PATH = 'greyhound_racing.db'
COMM = 0.05
FLAT_STAKE = 100

# Tracks to avoid
AVOID_TRACKS = ['Meadows (MEP)', 'Sale']

def load_data():
    """Load data with BSP prioritized"""
    print("Loading Data...")
    
    try:
        with open('tier1_tracks.txt', 'r') as f:
            safe_tracks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        safe_tracks = None
        
    conn = sqlite3.connect(DB_PATH)
    
    if safe_tracks:
        placeholders = ','.join('?' for _ in safe_tracks)
        track_filter = f"AND t.TrackName IN ({placeholders})"
        params = safe_tracks
    else:
        track_filter = ""
        params = []
    
    query = f"""
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        r.RaceNumber,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Position,
        ge.StartingPrice,
        ge.BSP
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND ge.BSP IS NOT NULL
      {track_filter}
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['RaceNumber'] = pd.to_numeric(df['RaceNumber'], errors='coerce').fillna(0).astype(int)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    # Use BSP directly (already filtered for NOT NULL)
    df['Odds'] = df['BSP']
    
    df['FieldSize'] = df.groupby('RaceID')['GreyhoundID'].transform('count')
    
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    print(f"Loaded {len(df)} entries with BSP data")
    return df.dropna(subset=['DogNormTimeAvg', 'Odds'])

def get_candidates(df, model, margin_threshold=0.1):
    """Get lay candidates using model predictions"""
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    df = df.copy()
    df['PredOverall'] = model.predict(df[features])
    df['PredRank'] = df.groupby('RaceID')['PredOverall'].rank(method='min')
    
    rank1s = df[df['PredRank'] == 1].copy()
    rank2s = df[df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    pool = candidates[
        (candidates['Margin'] > margin_threshold) &
        (candidates['Odds'] >= 1.50) &
        (candidates['Odds'] <= 3.00) &
        (~candidates['TrackName'].isin(AVOID_TRACKS))
    ].copy()
    
    return pool

def calculate_pnl(pool):
    """Calculate P&L for flat staking"""
    pool = pool.copy()
    pool['Stake'] = FLAT_STAKE
    pool['Liability'] = (pool['Odds'] - 1) * pool['Stake']
    pool['Profit'] = pool.apply(
        lambda r: r['Stake'] * (1 - COMM) if r['IsWin'] == 0 else -r['Liability'],
        axis=1
    )
    return pool

def calculate_drawdown(profits):
    """Calculate max drawdown from profit series"""
    cumulative = np.cumsum(profits)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return np.min(drawdown)

# ==============================================================================
# WALK-FORWARD TESTING
# ==============================================================================
def walk_forward_test(df, train_months=12, test_months=3):
    """Rolling window walk-forward validation"""
    print("\n" + "="*60)
    print("WALK-FORWARD TESTING")
    print("="*60)
    
    df = df.sort_values('MeetingDate')
    min_date = df['MeetingDate'].min()
    max_date = df['MeetingDate'].max()
    
    results = []
    current_start = min_date
    
    while current_start + pd.DateOffset(months=train_months + test_months) <= max_date:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        train_df = df[(df['MeetingDate'] >= current_start) & (df['MeetingDate'] < train_end)]
        test_df = df[(df['MeetingDate'] >= train_end) & (df['MeetingDate'] < test_end)]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            current_start += pd.DateOffset(months=test_months)
            continue
        
        # Train model
        features = ['DogNormTimeAvg', 'Box', 'Distance']
        model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, verbosity=0)
        model.fit(train_df[features], train_df['NormTime'])
        
        # Get candidates and calculate P&L
        pool = get_candidates(test_df, model)
        if len(pool) == 0:
            current_start += pd.DateOffset(months=test_months)
            continue
            
        pool = calculate_pnl(pool)
        
        period_profit = pool['Profit'].sum()
        period_bets = len(pool)
        period_wins = (pool['IsWin'] == 0).sum()
        
        results.append({
            'Period': f"{train_end.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
            'Bets': period_bets,
            'Wins': period_wins,
            'Strike': period_wins / period_bets * 100 if period_bets > 0 else 0,
            'Profit': period_profit
        })
        
        current_start += pd.DateOffset(months=test_months)
    
    print(f"\n{'Period':<25} {'Bets':<8} {'Strike':<10} {'Profit':<12}")
    print("-" * 55)
    for r in results:
        print(f"{r['Period']:<25} {r['Bets']:<8} {r['Strike']:.1f}%{'':<5} ${r['Profit']:>8,.0f}")
    
    total_profit = sum(r['Profit'] for r in results)
    total_bets = sum(r['Bets'] for r in results)
    print("-" * 55)
    print(f"{'TOTAL':<25} {total_bets:<8} {'':<10} ${total_profit:>8,.0f}")
    
    return results

# ==============================================================================
# K-FOLD CROSS VALIDATION
# ==============================================================================
def kfold_test(df, k=5):
    """K-Fold cross validation by time periods"""
    print("\n" + "="*60)
    print(f"{k}-FOLD CROSS VALIDATION")
    print("="*60)
    
    df = df.sort_values('MeetingDate')
    
    # Split into k time-based folds
    fold_size = len(df) // k
    results = []
    
    for fold in range(k):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < k - 1 else len(df)
        
        test_df = df.iloc[test_start:test_end].copy()
        train_df = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])
        
        if len(train_df) < 1000 or len(test_df) < 100:
            continue
        
        features = ['DogNormTimeAvg', 'Box', 'Distance']
        model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, verbosity=0)
        model.fit(train_df[features], train_df['NormTime'])
        
        pool = get_candidates(test_df, model)
        if len(pool) == 0:
            continue
            
        pool = calculate_pnl(pool)
        
        results.append({
            'Fold': fold + 1,
            'Bets': len(pool),
            'Strike': (pool['IsWin'] == 0).sum() / len(pool) * 100,
            'Profit': pool['Profit'].sum(),
            'MaxDD': calculate_drawdown(pool.sort_values('MeetingDate')['Profit'].values)
        })
    
    print(f"\n{'Fold':<8} {'Bets':<8} {'Strike':<10} {'Profit':<12} {'MaxDD':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['Fold']:<8} {r['Bets']:<8} {r['Strike']:.1f}%{'':<5} ${r['Profit']:>8,.0f} ${r['MaxDD']:>8,.0f}")
    
    avg_profit = np.mean([r['Profit'] for r in results])
    std_profit = np.std([r['Profit'] for r in results])
    print("-" * 50)
    print(f"Mean Profit: ${avg_profit:,.0f} (Std: ${std_profit:,.0f})")
    
    return results

# ==============================================================================
# MONTE CARLO SIMULATION
# ==============================================================================
def monte_carlo_test(df, n_simulations=1000):
    """Monte Carlo simulation with random bet ordering"""
    print("\n" + "="*60)
    print(f"MONTE CARLO SIMULATION ({n_simulations} runs)")
    print("="*60)
    
    # Train on earlier data, get candidates from later data
    cutoff = df['MeetingDate'].quantile(0.7)
    train_df = df[df['MeetingDate'] < cutoff]
    test_df = df[df['MeetingDate'] >= cutoff]
    
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, verbosity=0)
    model.fit(train_df[features], train_df['NormTime'])
    
    pool = get_candidates(test_df, model)
    pool = calculate_pnl(pool)
    
    profits = pool['Profit'].values
    
    # Run simulations with shuffled order
    final_profits = []
    max_drawdowns = []
    
    for _ in range(n_simulations):
        shuffled = np.random.permutation(profits)
        final_profits.append(np.sum(shuffled))
        max_drawdowns.append(calculate_drawdown(shuffled))
    
    final_profits = np.array(final_profits)
    max_drawdowns = np.array(max_drawdowns)
    
    print(f"\nBased on {len(profits)} actual bets:")
    print(f"Actual Total Profit: ${np.sum(profits):,.0f}")
    print(f"\nMonte Carlo Results:")
    print(f"Mean Final Profit:   ${np.mean(final_profits):,.0f}")
    print(f"Std Dev:             ${np.std(final_profits):,.0f}")
    print(f"5th Percentile:      ${np.percentile(final_profits, 5):,.0f}")
    print(f"95th Percentile:     ${np.percentile(final_profits, 95):,.0f}")
    print(f"\nDrawdown Analysis:")
    print(f"Mean Max Drawdown:   ${np.mean(max_drawdowns):,.0f}")
    print(f"Worst Max Drawdown:  ${np.min(max_drawdowns):,.0f}")
    print(f"95th %ile Drawdown:  ${np.percentile(max_drawdowns, 5):,.0f}")
    
    # Probability of profit
    prob_profit = (final_profits > 0).sum() / n_simulations * 100
    print(f"\nProbability of Profit: {prob_profit:.1f}%")
    
    return {
        'actual_profit': np.sum(profits),
        'mean_profit': np.mean(final_profits),
        'std_profit': np.std(final_profits),
        'prob_profit': prob_profit,
        'mean_drawdown': np.mean(max_drawdowns),
        'worst_drawdown': np.min(max_drawdowns)
    }

# ==============================================================================
# DRAWDOWN ANALYSIS
# ==============================================================================
def drawdown_analysis(df):
    """Detailed drawdown analysis"""
    print("\n" + "="*60)
    print("DRAWDOWN ANALYSIS")
    print("="*60)
    
    cutoff = df['MeetingDate'].quantile(0.7)
    train_df = df[df['MeetingDate'] < cutoff]
    test_df = df[df['MeetingDate'] >= cutoff]
    
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, verbosity=0)
    model.fit(train_df[features], train_df['NormTime'])
    
    pool = get_candidates(test_df, model)
    pool = calculate_pnl(pool)
    pool = pool.sort_values('MeetingDate')
    
    pool['CumProfit'] = pool['Profit'].cumsum()
    pool['CumMax'] = pool['CumProfit'].cummax()
    pool['Drawdown'] = pool['CumProfit'] - pool['CumMax']
    
    max_dd = pool['Drawdown'].min()
    max_dd_idx = pool['Drawdown'].idxmin()
    max_dd_date = pool.loc[max_dd_idx, 'MeetingDate']
    
    # Find recovery point
    peak_before_dd = pool.loc[:max_dd_idx, 'CumMax'].max()
    recovery_point = pool[pool['CumProfit'] >= peak_before_dd].index
    recovery_point = recovery_point[recovery_point > max_dd_idx]
    
    print(f"\nTotal Bets: {len(pool)}")
    print(f"Final Profit: ${pool['CumProfit'].iloc[-1]:,.0f}")
    print(f"\nMax Drawdown: ${max_dd:,.0f}")
    print(f"Max Drawdown Date: {max_dd_date.strftime('%Y-%m-%d')}")
    
    if len(recovery_point) > 0:
        recovery_date = pool.loc[recovery_point[0], 'MeetingDate']
        recovery_days = (recovery_date - max_dd_date).days
        print(f"Recovery Date: {recovery_date.strftime('%Y-%m-%d')} ({recovery_days} days)")
    else:
        print("Recovery: Not yet recovered")
    
    # Losing streaks
    pool['IsLoss'] = pool['Profit'] < 0
    losing_runs = []
    current_run = 0
    for loss in pool['IsLoss']:
        if loss:
            current_run += 1
        else:
            if current_run > 0:
                losing_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        losing_runs.append(current_run)
    
    print(f"\nMax Losing Streak: {max(losing_runs) if losing_runs else 0} bets")
    print(f"Avg Losing Streak: {np.mean(losing_runs):.1f} bets" if losing_runs else "")
    
    return {
        'max_drawdown': max_dd,
        'final_profit': pool['CumProfit'].iloc[-1],
        'max_losing_streak': max(losing_runs) if losing_runs else 0
    }

def main():
    print("="*60)
    print("DEEP TESTING: FLAT STAKING @ BSP")
    print("="*60)
    
    df = load_data()
    
    # Run all tests
    wf_results = walk_forward_test(df)
    kf_results = kfold_test(df)
    mc_results = monte_carlo_test(df)
    dd_results = drawdown_analysis(df)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Walk-Forward Total Profit: ${sum(r['Profit'] for r in wf_results):,.0f}")
    print(f"K-Fold Mean Profit:        ${np.mean([r['Profit'] for r in kf_results]):,.0f}")
    print(f"Monte Carlo Prob of Profit: {mc_results['prob_profit']:.1f}%")
    print(f"Max Drawdown:               ${dd_results['max_drawdown']:,.0f}")

if __name__ == "__main__":
    main()
