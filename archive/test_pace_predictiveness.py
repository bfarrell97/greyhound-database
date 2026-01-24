"""
Test: Do dogs with GOOD HISTORICAL PACE actually win more?
This validates if LastN_AvgFinishBenchmark is truly predictive
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def test_historical_pace_predictiveness():
    """Test if historical pace predicts future wins"""
    print("\n" + "="*80)
    print("TESTING HISTORICAL PACE PREDICTIVENESS")
    print("="*80)
    print("""
This test answers: Do dogs with good historical pace actually win more races?
We're looking at PAST races to build pace history, then measuring if that
predicts FUTURE race outcomes.

Metrics:
  - Historical FinishBenchmark (average of last 5 races)
  - Quartile analysis: Is there a monotonic relationship?
""")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get all races with historical data available
    query = """
    WITH dog_pace_history AS (
        SELECT 
            ge.GreyhoundID,
            g.GreyhoundName,
            rm.MeetingDate,
            (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
    ),
    
    dog_pace_avg AS (
        SELECT 
            GreyhoundID,
            GreyhoundName,
            AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg,
            COUNT(DISTINCT CASE WHEN RaceNum <= 5 THEN MeetingDate END) as RacesUsed
        FROM dog_pace_history
        GROUP BY GreyhoundID
        HAVING RacesUsed >= 5
    ),
    
    future_races AS (
        SELECT 
            ge.GreyhoundID,
            (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
            ge.StartingPrice,
            dpa.HistoricalPaceAvg,
            ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
        FROM GreyhoundEntries ge
        JOIN dog_pace_avg dpa ON ge.GreyhoundID = dpa.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.Position IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR')
          AND ge.StartingPrice IS NOT NULL
          AND rm.MeetingDate >= '2025-01-01'
    )
    
    SELECT * FROM future_races WHERE RaceNum > 1
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nLoaded {len(df):,} dogs with 5+ races of history and future race outcomes")
    
    # Convert data types
    df['IsWinner'] = pd.to_numeric(df['IsWinner'], errors='coerce').fillna(0)
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['HistoricalPaceAvg'] = pd.to_numeric(df['HistoricalPaceAvg'], errors='coerce')
    
    # Remove rows with missing data
    df = df.dropna(subset=['IsWinner', 'HistoricalPaceAvg', 'StartingPrice'])
    
    print(f"After cleaning: {len(df):,} races with complete data")
    print(f"Overall win rate: {df['IsWinner'].mean()*100:.2f}%\n")
    
    # TEST 1: Quartile analysis
    print("="*80)
    print("TEST 1: QUARTILE ANALYSIS")
    print("="*80)
    print("Split dogs into 4 groups by historical pace, check if higher pace = higher win rate")
    
    df['PaceQuartile'] = pd.qcut(df['HistoricalPaceAvg'], q=4, duplicates='drop')
    
    for quartile in sorted(df['PaceQuartile'].unique()):
        q_data = df[df['PaceQuartile'] == quartile]
        wins = q_data['IsWinner'].sum()
        total = len(q_data)
        win_rate = wins / total * 100
        
        avg_pace = q_data['HistoricalPaceAvg'].mean()
        
        print(f"\n{quartile}:")
        print(f"  Dogs: {total:,}")
        print(f"  Avg Historical Pace: {avg_pace:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # TEST 2: Threshold analysis
    print("\n" + "="*80)
    print("TEST 2: THRESHOLD ANALYSIS")
    print("="*80)
    print("Do dogs with above-threshold historical pace win more?")
    
    for threshold in [-1.0, -0.5, 0, 0.5, 1.0, 1.5]:
        above_threshold = df[df['HistoricalPaceAvg'] >= threshold]
        
        if len(above_threshold) < 100:
            continue
        
        wins = above_threshold['IsWinner'].sum()
        total = len(above_threshold)
        win_rate = wins / total * 100
        
        # Calculate ROI on $1.50-$2.00
        in_odds = above_threshold[
            (above_threshold['StartingPrice'] >= 1.50) &
            (above_threshold['StartingPrice'] <= 2.00)
        ]
        
        if len(in_odds) > 0:
            in_odds_wins = in_odds['IsWinner'].sum()
            in_odds_total = len(in_odds)
            in_odds_strike = in_odds_wins / in_odds_total * 100
            
            # ROI
            stakes = in_odds_total * 1.0
            returns = (in_odds[in_odds['IsWinner'] == 1]['StartingPrice'].sum())
            roi = ((returns - stakes) / stakes * 100) if stakes > 0 else 0
            
            print(f"\nHistorical Pace >= {threshold:>4}:")
            print(f"  All races: {total:,} dogs, {win_rate:.1f}% win rate")
            print(f"  $1.50-$2.00: {in_odds_total:,} bets, {in_odds_strike:.1f}% strike, ROI: {roi:+.2f}%")
    
    # TEST 3: Correlation
    print("\n" + "="*80)
    print("TEST 3: CORRELATION ANALYSIS")
    print("="*80)
    
    correlation = df['HistoricalPaceAvg'].corr(df['IsWinner'])
    print(f"Correlation (HistoricalPace → Win): {correlation:.4f}")
    
    if correlation > 0.1:
        print("✓ POSITIVE correlation - historical pace DOES predict wins")
    elif correlation > -0.1:
        print("≈ Weak correlation - historical pace has limited predictive power")
    else:
        print("✗ NEGATIVE correlation - historical pace DOES NOT predict wins (opposite effect)")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if df[df['HistoricalPaceAvg'] >= df['HistoricalPaceAvg'].quantile(0.75)]['IsWinner'].mean() > df['IsWinner'].mean():
        print("✓ Dogs with above-average historical pace WIN MORE")
        print("✓ LastN_AvgFinishBenchmark IS a predictive feature")
    else:
        print("✗ Dogs with above-average historical pace do NOT win more consistently")
        print("✗ LastN_AvgFinishBenchmark may NOT be predictive in live betting")
        print("\nThis could mean:")
        print("  1. The feature had chance correlation during training")
        print("  2. Live betting patterns differ from training data")
        print("  3. Need to combine with other signals (Box, Track, Class)")

if __name__ == "__main__":
    test_historical_pace_predictiveness()
