"""Test Backing Strategy at BSP < $8"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'
COMM = 0.05  # 5% Betfair commission

def load_data():
    print("Loading data with BSP...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT ge.GreyhoundID, g.GreyhoundName, r.RaceID, r.RaceNumber,
           rm.MeetingDate, t.TrackName, r.Distance, ge.Box, ge.Position, ge.BSP,
           ge.FinishTime, ge.FirstSplitPosition, ge.FinishTimeBenchmarkLengths,
           rm.MeetingAvgBenchmarkLengths, ge.CareerPrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2023-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.BSP IS NOT NULL
      AND ge.BSP < 8
    ORDER BY rm.MeetingDate, r.RaceID, ge.Box
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} entries with BSP < $8")
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
    df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
    df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce').fillna(0)
    df['TotalPace'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']
    
    return df

def calculate_features(df):
    print("Calculating historical features...")
    df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
    
    df['HistAvgSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    df['HistAvgPace'] = df.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    df['PriorRaces'] = df.groupby('GreyhoundID').cumcount()
    
    return df

def identify_leaders(df):
    print("Identifying race leaders...")
    
    BOX_ADJ = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(BOX_ADJ).fillna(0)
    df['PredictedSplit'] = df['HistAvgSplit'] + df['BoxAdj']
    
    df['PaceRank'] = df.groupby('RaceID')['HistAvgPace'].rank(method='min', ascending=False)
    df['PIRRank'] = df.groupby('RaceID')['PredictedSplit'].rank(method='min', ascending=True)
    
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPIRLeader'] = df['PIRRank'] == 1
    df['IsBothLeader'] = df['IsPaceLeader'] & df['IsPIRLeader']
    
    return df

def backtest(df, name, filter_col, require_split=True, require_pace=True):
    filter_cond = df[filter_col]
    if require_pace:
        filter_cond = filter_cond & df['HistAvgPace'].notna()
    if require_split:
        filter_cond = filter_cond & df['HistAvgSplit'].notna()
    
    pool = df[filter_cond].copy()
    if len(pool) == 0:
        return None
    
    pool['Stake'] = 10
    pool['Profit'] = pool.apply(
        lambda r: r['Stake'] * (r['BSP'] - 1) * (1 - COMM) if r['IsWin'] == 1 else -r['Stake'], axis=1
    )
    
    total_profit = pool['Profit'].sum()
    total_stake = pool['Stake'].sum()
    wins = pool['IsWin'].sum()
    strike = wins / len(pool) * 100
    roi = total_profit / total_stake * 100
    avg_bsp = pool['BSP'].mean()
    
    pool['Year'] = pool['MeetingDate'].dt.year
    yearly = pool.groupby('Year')['Profit'].sum().to_dict()
    
    return {
        'Strategy': name, 'Bets': len(pool), 'Wins': wins, 'Strike': strike,
        'AvgBSP': avg_bsp, 'Profit': total_profit, 'ROI': roi, 'Yearly': yearly
    }

def main():
    print("="*70)
    print("BACKING STRATEGIES TEST - BSP < $8")
    print("="*70)
    
    df = load_data()
    df = calculate_features(df)
    df = identify_leaders(df)
    df = df[df['PriorRaces'] >= 5].copy()
    
    print(f"\nDogs with 5+ prior races: {len(df):,}")
    
    configs = [
        ('Pace Leader', 'IsPaceLeader', False, True),
        ('PIR Leader', 'IsPIRLeader', True, False),
        ('Pace+PIR', 'IsBothLeader', True, True),
    ]
    
    print("\n" + "="*70)
    print("RESULTS (All BSP < $8)")
    print("="*70)
    
    results = []
    for name, filter_col, req_split, req_pace in configs:
        result = backtest(df, name, filter_col, require_split=req_split, require_pace=req_pace)
        if result:
            results.append(result)
    
    print(f"\n{'Strategy':<15} {'Bets':<8} {'Strike':<10} {'AvgBSP':<10} {'Profit':<12} {'ROI':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['Strategy']:<15} {r['Bets']:<8} {r['Strike']:.1f}%{'':<5} ${r['AvgBSP']:<9.2f} ${r['Profit']:>+10,.0f} {r['ROI']:>+7.1f}%")
    
    print("\n" + "="*70)
    print("YEARLY BREAKDOWN")
    print("="*70)
    
    years = [2023, 2024, 2025]
    print(f"\n{'Strategy':<15}", end="")
    for y in years:
        print(f" {y:<12}", end="")
    print()
    print("-" * 55)
    
    for r in results:
        print(f"{r['Strategy']:<15}", end="")
        for y in years:
            profit = r['Yearly'].get(y, 0)
            print(f" ${profit:>+10,.0f}", end="")
        print()

if __name__ == "__main__":
    main()
