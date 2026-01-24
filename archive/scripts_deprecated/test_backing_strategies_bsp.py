"""
Test Backing Strategies at BSP
==============================
Tests three backing strategies using BSP prices:
1. Pace Leader Only (fastest average finish vs benchmark)
2. PIR Leader Only (predicted to lead at first split)  
3. Combined Pace + PIR Leader

Uses walk-forward testing for robust results.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = 'greyhound_racing.db'
COMM = 0.05  # 5% Betfair commission on winnings

def load_data():
    """Load race data with BSP and historical features"""
    print("Loading data with BSP...")
    
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        r.RaceNumber,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Position,
        ge.BSP,
        ge.StartingPrice,
        ge.FinishTime,
        ge.FirstSplitPosition,
        ge.FinishTimeBenchmarkLengths,
        rm.MeetingAvgBenchmarkLengths,
        ge.CareerPrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2023-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.BSP IS NOT NULL
    ORDER BY rm.MeetingDate, r.RaceID, ge.Box
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} entries with BSP")
    
    # Data cleaning
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
    df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
    df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
    df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
    
    # Calculate TotalPace (higher = better)
    df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']
    
    return df

def calculate_historical_features(df):
    """Calculate historical averages for each dog before each race"""
    print("Calculating historical features...")
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
    
    # Historical average first split position (lower is better - led more)
    df['HistAvgSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    # Historical average pace (higher is better - faster)
    df['HistAvgPace'] = df.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    # Count of prior races
    df['PriorRaces'] = df.groupby('GreyhoundID').cumcount()
    
    print(f"  Dogs with historical split data: {df['HistAvgSplit'].notna().sum():,}")
    print(f"  Dogs with historical pace data: {df['HistAvgPace'].notna().sum():,}")
    
    return df

def identify_leaders(df):
    """Identify Pace Leaders and PIR Leaders within each race"""
    print("Identifying race leaders...")
    
    # Box adjustment for PIR prediction
    BOX_ADJ = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(BOX_ADJ).fillna(0)
    df['PredictedSplit'] = df['HistAvgSplit'] + df['BoxAdj']
    
    # Rank within each race
    # Pace: Higher is better (ascending=False means rank 1 = highest pace)
    df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=False)
    
    # PIR: Lower predicted split is better (ascending=True means rank 1 = lowest split = leads)
    df['PIRRank'] = df.groupby('RaceID')['PredictedSplit'].rank(method='min', ascending=True)
    
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPIRLeader'] = df['PIRRank'] == 1
    df['IsBothLeader'] = df['IsPaceLeader'] & df['IsPIRLeader']
    
    # Field size
    df['FieldSize'] = df.groupby('RaceID')['GreyhoundID'].transform('count')
    
    print(f"  Pace Leaders: {df['IsPaceLeader'].sum():,}")
    print(f"  PIR Leaders: {df['IsPIRLeader'].sum():,}")
    print(f"  Both Leaders: {df['IsBothLeader'].sum():,}")
    
    return df

def backtest_strategy(df, strategy_name, filter_col, odds_min=1.0, odds_max=100.0, flat_stake=10, require_split=True, require_pace=True):
    """Backtest a backing strategy"""
    
    # Build filter based on requirements
    filter_cond = df[filter_col] & (df['BSP'] >= odds_min) & (df['BSP'] <= odds_max)
    
    if require_pace:
        filter_cond = filter_cond & df['HistAvgPace'].notna()
    if require_split:
        filter_cond = filter_cond & df['HistAvgSplit'].notna()
    
    pool = df[filter_cond].copy()
    
    if len(pool) == 0:
        return None
    
    pool['Stake'] = flat_stake
    
    # Backing P&L: Win = Stake * (BSP - 1) * (1 - commission), Lose = -Stake
    pool['Profit'] = pool.apply(
        lambda r: r['Stake'] * (r['BSP'] - 1) * (1 - COMM) if r['IsWin'] == 1 else -r['Stake'],
        axis=1
    )
    
    total_profit = pool['Profit'].sum()
    total_stake = pool['Stake'].sum()
    total_bets = len(pool)
    wins = pool['IsWin'].sum()
    strike = wins / total_bets * 100 if total_bets > 0 else 0
    roi = total_profit / total_stake * 100 if total_stake > 0 else 0
    
    # Drawdown
    pool = pool.sort_values('MeetingDate')
    pool['CumProfit'] = pool['Profit'].cumsum()
    pool['CumMax'] = pool['CumProfit'].cummax()
    pool['Drawdown'] = pool['CumProfit'] - pool['CumMax']
    max_dd = pool['Drawdown'].min()
    
    # Yearly breakdown
    pool['Year'] = pool['MeetingDate'].dt.year
    yearly = pool.groupby('Year')['Profit'].sum().to_dict()
    
    # Average BSP of selections
    avg_bsp = pool['BSP'].mean()
    
    return {
        'Strategy': strategy_name,
        'OddsRange': f"${odds_min:.0f}-${odds_max:.0f}",
        'Bets': total_bets,
        'Wins': wins,
        'Strike': strike,
        'AvgBSP': avg_bsp,
        'TotalStake': total_stake,
        'TotalProfit': total_profit,
        'ROI': roi,
        'MaxDD': max_dd,
        'Yearly': yearly
    }

def main():
    print("="*70)
    print("BACKING STRATEGIES TEST AT BSP")
    print("="*70)
    
    df = load_data()
    df = calculate_historical_features(df)
    df = identify_leaders(df)
    
    # Filter to dogs with sufficient history
    df = df[df['PriorRaces'] >= 5].copy()
    print(f"\nDogs with 5+ prior races: {len(df):,}")
    
    # Define test configurations: (name, filter_col, odds_min, odds_max, require_split, require_pace)
    configs = [
        # Pace Leader (only needs pace data)
        ('Pace Leader', 'IsPaceLeader', 1.0, 100.0, False, True),
        ('Pace Leader Favs', 'IsPaceLeader', 1.5, 3.0, False, True),
        ('Pace Leader Mid', 'IsPaceLeader', 3.0, 10.0, False, True),
        ('Pace Leader Long', 'IsPaceLeader', 10.0, 50.0, False, True),
        
        # PIR Leader (needs split data)
        ('PIR Leader', 'IsPIRLeader', 1.0, 100.0, True, False),
        ('PIR Leader Favs', 'IsPIRLeader', 1.5, 3.0, True, False),
        ('PIR Leader Mid', 'IsPIRLeader', 3.0, 10.0, True, False),
        ('PIR Leader Long', 'IsPIRLeader', 10.0, 50.0, True, False),
        
        # Combined (needs both)
        ('Pace+PIR', 'IsBothLeader', 1.0, 100.0, True, True),
        ('Pace+PIR Favs', 'IsBothLeader', 1.5, 3.0, True, True),
        ('Pace+PIR Mid', 'IsBothLeader', 3.0, 10.0, True, True),
        ('Pace+PIR Long', 'IsBothLeader', 10.0, 50.0, True, True),
    ]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    results = []
    for name, filter_col, odds_min, odds_max, req_split, req_pace in configs:
        result = backtest_strategy(df, name, filter_col, odds_min, odds_max, require_split=req_split, require_pace=req_pace)
        if result:
            results.append(result)
    
    # Print results table
    print(f"\n{'Strategy':<18} {'Odds':<12} {'Bets':<7} {'Strike':<8} {'AvgBSP':<8} {'Profit':<12} {'ROI':<8} {'MaxDD':<10}")
    print("-" * 95)
    
    for r in results:
        profit_str = f"${r['TotalProfit']:+,.0f}"
        roi_str = f"{r['ROI']:+.1f}%"
        maxdd_str = f"${r['MaxDD']:,.0f}"
        print(f"{r['Strategy']:<18} {r['OddsRange']:<12} {r['Bets']:<7} {r['Strike']:.1f}%{'':<3} ${r['AvgBSP']:<7.2f} {profit_str:<12} {roi_str:<8} {maxdd_str:<10}")
    
    # Find best strategies
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    profitable = [r for r in results if r['ROI'] > 0]
    if profitable:
        best_roi = max(profitable, key=lambda x: x['ROI'])
        best_profit = max(profitable, key=lambda x: x['TotalProfit'])
        
        print(f"\nBest ROI: {best_roi['Strategy']} @ {best_roi['OddsRange']} = {best_roi['ROI']:+.1f}%")
        print(f"Best Total Profit: {best_profit['Strategy']} @ {best_profit['OddsRange']} = ${best_profit['TotalProfit']:+,.0f}")
    elif results:
        print("\n*** NO PROFITABLE STRATEGIES FOUND AT BSP ***")
        least_bad = min(results, key=lambda x: abs(x['ROI']))
        print(f"Least unprofitable: {least_bad['Strategy']} @ {least_bad['OddsRange']} = {least_bad['ROI']:+.1f}%")
    else:
        print("\n*** NO RESULTS - Check data filters ***")
    
    # Yearly breakdown for top strategies
    print("\n" + "="*70)
    print("YEARLY BREAKDOWN (Top 3 by ROI)")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda x: x['ROI'], reverse=True)[:3]
    years = [2023, 2024, 2025]
    
    print(f"\n{'Strategy':<25}", end="")
    for y in years:
        print(f" {y:<12}", end="")
    print()
    print("-" * 65)
    
    for r in sorted_results:
        print(f"{r['Strategy']} ({r['OddsRange']})"[:25].ljust(25), end="")
        for y in years:
            profit = r['Yearly'].get(y, 0)
            print(f" ${profit:>+10,.0f}", end="")
        print()

if __name__ == "__main__":
    main()
