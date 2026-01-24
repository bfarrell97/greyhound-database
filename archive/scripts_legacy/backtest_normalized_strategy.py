"""
Backtest Normalized Split Strategy
Implements the "Normalized Split" logic to fix track/distance mixing errors.

Methodology:
1. Calculate Benchmark Split (Median) for every Track/Distance.
2. For each race, calculate NormSplit = ActualSplit - Benchmark.
3. Calculate RollingAvgNormSplit (Last 5 races).
4. Predict Today's Split = RollingAvgNormSplit + TodayBenchmark.
5. Apply Standard Filters (<=600m, >=6 Dogs, Fav>=$1.30).
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
start_date = '2020-01-01'
end_date = '2025-12-09'
min_odds = 1.50
max_odds = 30.0
DB_PATH = 'greyhound_racing.db'

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_normalized_backtest():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. LOAD ALL DATA
    progress("Loading race data...")
    query = """
    SELECT
        rm.MeetingID,
        rm.MeetingDate,
        t.TrackName,
        r.RaceNumber,
        r.Distance,
        ge.GreyhoundID,
        g.GreyhoundName,
        ge.Box,
        ge.Split,
        ge.StartingPrice as CurrentOdds,
        ge.CareerPrizeMoney,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        (ge.FinishTimeBenchmarkLengths + COALESCE(rm.MeetingAvgBenchmarkLengths, 0)) as TotalPace,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2019-01-01' -- Load extra history for rolling calcs
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    
    # Clean
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    df['CurrentOdds'] = pd.to_numeric(df['CurrentOdds'], errors='coerce')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['RaceKey'] = df['MeetingID'].astype(str) + '_R' + df['RaceNumber'].astype(str)
    
    progress(f"Loaded {len(df):,} records")

    # 2. CALCULATE BENCHMARKS (Track/Distance Medians)
    progress("Calculating benchmarks...")
    # Only use valid splits within reasonable range
    valid_splits = df[(df['Split'] > 0) & (df['Split'] < 30)]
    benchmarks = valid_splits.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
    
    # Merge benchmarks
    df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
    
    # Calculate NormSplit
    df['NormSplit'] = df['Split'] - df['TrackDistMedian']
    
    # 3. ROLLING CALCULATIONS
    progress("Computing rolling normalized stats...")
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Rolling Avg NormSplit
    # We shift(1) so we only use PAST races
    df['RollingNormSplit'] = df.groupby('GreyhoundID')['NormSplit'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Track the date of the newest split used (to check freshness)
    df['PrevRaceDate'] = df.groupby('GreyhoundID')['MeetingDate'].shift(1)
    
    # Rolling Avg Pace (Standard)
    df['RollingPace'] = df.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Running Prize Money
    df['CumPrize'] = df.groupby('GreyhoundID')['PrizeMoney'].cumsum()
    df['RunningPrize'] = df.groupby('GreyhoundID')['CumPrize'].shift(1).fillna(0)
    
    # 4. FILTER TO BACKTEST WINDOW & APPLY STRATEGY
    progress("Applying strategy logic...")
    
    # Filter to Backtest Window
    bt_df = df[(df['MeetingDate'] >= start_date) & (df['MeetingDate'] <= end_date)].copy()
    
    # Market Stats (Overround/MinOdds)
    # Note: We need to calc this on the filtered set or pre-calc. 
    # Let's simple-calc on bt_df, assuming we have all runners for these races.
    bt_df['ImpliedProb'] = 1.0 / bt_df['CurrentOdds'].replace(0, np.nan)
    market_stats = bt_df.groupby('RaceKey').agg({
        'ImpliedProb': 'sum',
        'CurrentOdds': 'min',
        'GreyhoundID': 'count'
    }).rename(columns={
        'ImpliedProb': 'MarketOverround', 
        'CurrentOdds': 'MinMarketOdds',
        'GreyhoundID': 'FieldSize'
    }).reset_index()
    
    bt_df = bt_df.merge(market_stats, on='RaceKey', how='left')
    
    # Apply Standard Filters
    # Note: NZ/TAS are already purged from DB
    filters = (
        (bt_df['Distance'] <= 600) &
        (bt_df['FieldSize'] >= 6) &
        (bt_df['MinMarketOdds'] >= 1.30) &
        (bt_df['MarketOverround'] <= 1.40) &
        (bt_df['RollingNormSplit'].notna())
    )
    strategy_df = bt_df[filters].copy()
    
    # PREDICT SPLIT
    # Predicted = RollingNorm + TodayBenchmark + BoxAdj
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    strategy_df['BoxAdj'] = strategy_df['Box'].map(box_adj).fillna(0)
    
    strategy_df['PredictedSplit'] = (
        strategy_df['RollingNormSplit'] + 
        strategy_df['TrackDistMedian'] + 
        strategy_df['BoxAdj']
    )
    
    # Ranks
    strategy_df['PredictedSplitRank'] = strategy_df.groupby('RaceKey')['PredictedSplit'].rank(method='min')
    strategy_df['PaceRank'] = strategy_df.groupby('RaceKey')['RollingPace'].rank(method='min', ascending=True)
    
    # Logic
    strategy_df['IsPIRLeader'] = strategy_df['PredictedSplitRank'] == 1
    strategy_df['IsPaceLeader'] = strategy_df['PaceRank'] == 1
    strategy_df['HasMoney'] = strategy_df['RunningPrize'] >= 30000
    strategy_df['InOddsRange'] = (strategy_df['CurrentOdds'] >= min_odds) & (strategy_df['CurrentOdds'] <= max_odds)
    
    # Data Age Check
    strategy_df['DaysSinceLastRace'] = (strategy_df['MeetingDate'] - strategy_df['PrevRaceDate']).dt.days
    
    # 5. RESULTS
    # Strategy: Normalized Leader
    mask = (
        strategy_df['IsPIRLeader'] & 
        strategy_df['IsPaceLeader'] & 
        strategy_df['HasMoney'] & 
        strategy_df['InOddsRange']
    )
    bets = strategy_df[mask].copy()
    
    # Staking
    def get_stake(odds):
        if odds < 3: return 0.5
        elif odds < 5: return 0.75
        elif odds < 10: return 1.0
        elif odds < 20: return 1.5
        else: return 2.0
        
    bets['Stake'] = bets['CurrentOdds'].apply(get_stake)
    bets['Return'] = bets.apply(lambda r: r['Stake'] * r['CurrentOdds'] if r['IsWinner'] else 0, axis=1)
    bets['Profit'] = bets['Return'] - bets['Stake']
    
    # Report
    print("\n" + "="*80)
    print("NORMALIZED STRATEGY BACKTEST (CLEAN DB)")
    print("Method: Rolling Normalized Splits (Z-Score approach equivalent)")
    print("Filters: <=600m, >=6 Dogs, Fav>=$1.30")
    print("="*80)
    
    count = len(bets)
    if count > 0:
        win_rate = bets['IsWinner'].mean() * 100
        profit = bets['Profit'].sum()
        roi = (profit / bets['Stake'].sum()) * 100
        print(f"Total Bets: {count:,}")
        print(f"Win Rate:   {win_rate:.1f}%")
        print(f"Profit:     {profit:.2f}u")
        print(f"ROI:        {roi:.1f}%")
        
        print("\n--- Freshness Check (Days Since Last Race) ---")
        fresh = bets[bets['DaysSinceLastRace'] <= 30]
        stale = bets[bets['DaysSinceLastRace'] > 90]
        
        if len(fresh) > 0:
            fresh_roi = (fresh['Profit'].sum() / fresh['Stake'].sum()) * 100
            print(f"Fresh (<30 days): {len(fresh)} bets, {fresh_roi:.1f}% ROI")
        
        if len(stale) > 0:
            stale_roi = (stale['Profit'].sum() / stale['Stake'].sum()) * 100
            print(f"Stale (>90 days): {len(stale)} bets, {stale_roi:.1f}% ROI")
            
        print("\n--- Yearly Breakdown ---")
        bets['Year'] = bets['MeetingDate'].dt.year
        yearly = bets.groupby('Year').agg({
            'IsWinner': 'count',
            'Profit': 'sum',
            'Stake': 'sum'
        })
        yearly['ROI'] = yearly['Profit'] / yearly['Stake'] * 100
        print(yearly[['IsWinner', 'Profit', 'ROI']])
        
    else:
        print("No bets found.")
        
    conn.close()

if __name__ == "__main__":
    run_normalized_backtest()
