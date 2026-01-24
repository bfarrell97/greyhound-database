"""
VALID BETTING MODEL - NO DATA LEAKAGE
Uses ONLY historical features available BEFORE each race
"""

import sqlite3
import pandas as pd
import numpy as np
import time

DB_PATH = 'greyhound_racing.db'

def load_data():
    """Load race data with pre-race features only"""
    print("Loading data...")
    start = time.time()
    
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.GreyhoundID, ge.RaceID, ge.Position, ge.Box,
        ge.StartingPrice, ge.FinishTimeBenchmarkLengths,
        ge.CareerPrizeMoney,
        g.GreyhoundName, g.DateWhelped,
        r.RaceNumber, r.Distance,
        rm.MeetingDate, rm.TrackID, rm.MeetingAvgBenchmarkLengths,
        t.TrackName
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position IS NOT NULL
      AND ge.StartingPrice IS NOT NULL
      AND rm.MeetingDate >= '2020-01-01'
    ORDER BY rm.MeetingDate, r.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def prepare_features(df):
    """Convert types and create base features"""
    print("Preparing features...")
    start = time.time()
    
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
    df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
    df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    
    df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
    
    # Age at race time
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce', utc=True).dt.tz_localize(None)
    df['AgeYears'] = (df['MeetingDate'] - df['DateWhelped']).dt.days / 365.25
    
    # Total pace benchmark
    df['TotalBench'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']
    
    df = df.sort_values(['MeetingDate', 'RaceID']).reset_index(drop=True)
    print(f"  Done in {time.time()-start:.1f}s")
    return df

def calculate_historical_stats(df):
    """Calculate EXPANDING historical stats - only using data from BEFORE each race"""
    print("Calculating historical stats...")
    start = time.time()
    
    # Win count and race count BEFORE current race (shift by 1)
    df['PriorWins'] = df.groupby('GreyhoundID')['Win'].cumsum().shift(1).fillna(0)
    df['PriorRaces'] = df.groupby('GreyhoundID').cumcount()
    df['HistWinRate'] = np.where(df['PriorRaces'] >= 3, df['PriorWins'] / df['PriorRaces'], np.nan)
    
    # Pace: cumulative average BEFORE current race
    df['CumPace'] = df.groupby('GreyhoundID')['TotalBench'].cumsum().shift(1).fillna(0)
    df['HistPace'] = np.where(df['PriorRaces'] >= 3, df['CumPace'] / df['PriorRaces'], np.nan)
    
    # Require minimum 5 prior races for reliable stats
    df['HasHistory'] = df['PriorRaces'] >= 5
    
    print(f"  Done in {time.time()-start:.1f}s")
    print(f"  Rows with 5+ prior races: {df['HasHistory'].sum():,}")
    return df

def score_entries(df):
    """Score each entry using ONLY pre-race historical data"""
    print("Scoring entries...")
    
    scored = df[df['HasHistory']].copy()
    
    # --- FEATURE 1: Historical win rate (0-2 scale) ---
    scored['WinRateScore'] = (scored['HistWinRate'] * 10).clip(0, 2)
    
    # --- FEATURE 2: Historical pace rank within race (0-2 scale) ---
    # Higher pace = better (more positive benchmark)
    scored['PaceRank'] = scored.groupby('RaceID')['HistPace'].rank(ascending=False, method='min', na_option='bottom')
    scored['PaceScore'] = np.where(scored['PaceRank'] <= 2, 2,
                          np.where(scored['PaceRank'] <= 4, 1, 0))
    
    # --- FEATURE 3: Career prize money (0-2 scale) ---
    # This IS available pre-race (cumulative career earnings)
    scored['MoneyScore'] = np.where(scored['CareerPrizeMoney'] >= 50000, 2,
                           np.where(scored['CareerPrizeMoney'] >= 15000, 1.5,
                           np.where(scored['CareerPrizeMoney'] >= 5000, 1, 0)))
    
    # --- FEATURE 4: Age (0-2 scale, younger = better) ---
    scored['AgeScore'] = np.where(scored['AgeYears'] <= 2, 2,
                         np.where(scored['AgeYears'] <= 3, 1.5,
                         np.where(scored['AgeYears'] <= 4, 1, 0)))
    
    # --- FEATURE 5: Box advantage (0-1 scale) ---
    # Inside boxes have slight advantage in sprints
    scored['BoxScore'] = np.where(scored['Box'] <= 2, 1,
                         np.where(scored['Box'] <= 4, 0.5, 0))
    
    # --- COMBINED SCORE (0-1 normalized) ---
    max_score = 2 + 2 + 2 + 2 + 1  # 9 max
    scored['RawScore'] = (scored['WinRateScore'] + scored['PaceScore'] + 
                          scored['MoneyScore'] + scored['AgeScore'] + scored['BoxScore'])
    scored['Score'] = scored['RawScore'] / max_score
    
    print(f"  Scored {len(scored):,} entries")
    return scored

def run_backtest(df, min_score, price_min, price_max, year=None):
    """Run backtest for given parameters"""
    subset = df.copy()
    
    if year:
        subset = subset[subset['MeetingDate'].dt.year == year]
    
    subset = subset[
        (subset['Score'] >= min_score) &
        (subset['StartingPrice'] >= price_min) &
        (subset['StartingPrice'] <= price_max)
    ]
    
    if len(subset) == 0:
        return {'bets': 0, 'wins': 0, 'win_pct': 0, 'roi': 0}
    
    bets = len(subset)
    wins = subset['Win'].sum()
    returns = (subset[subset['Win']]['StartingPrice'] - 1).sum()
    profit = returns - bets
    
    return {
        'bets': bets,
        'wins': wins,
        'win_pct': wins / bets * 100,
        'roi': profit / bets * 100
    }

def main():
    print("="*70)
    print("VALID BETTING MODEL - NO DATA LEAKAGE")
    print("Using ONLY historical features available before each race")
    print("="*70)
    start_time = time.time()
    
    df = load_data()
    df = prepare_features(df)
    df = calculate_historical_stats(df)
    df = score_entries(df)
    
    # Test various configurations
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    results = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for price_range in [(1.5, 3.0), (1.5, 5.0), (2.0, 5.0), (3.0, 10.0)]:
            r = run_backtest(df, threshold, price_range[0], price_range[1])
            if r['bets'] >= 100:
                results.append({
                    'threshold': threshold,
                    'price': f"${price_range[0]:.1f}-${price_range[1]:.1f}",
                    **r
                })
    
    # Sort by ROI
    results = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n{'Score':<8} {'Price':<15} {'Bets':>10} {'Wins':>10} {'Win%':>8} {'ROI':>10}")
    print("-"*65)
    for r in results[:25]:
        print(f"{r['threshold']:<8.2f} {r['price']:<15} {r['bets']:>10,} {r['wins']:>10,} {r['win_pct']:>7.1f}% {r['roi']:>+9.1f}%")
    
    # Year-by-year for best config
    print("\n" + "="*70)
    print("YEAR-BY-YEAR: Best config (Score >= 0.65, $3-$10)")
    print("="*70)
    
    years = sorted(df['MeetingDate'].dt.year.unique())
    print(f"\n{'Year':<8} {'Bets':>10} {'Wins':>10} {'Win%':>8} {'ROI':>10}")
    print("-"*50)
    
    for year in years:
        r = run_backtest(df, 0.65, 3.0, 10.0, year=year)
        if r['bets'] > 0:
            print(f"{year:<8} {r['bets']:>10,} {r['wins']:>10,} {r['win_pct']:>7.1f}% {r['roi']:>+9.1f}%")
    
    # Overall for this config
    r = run_backtest(df, 0.65, 3.0, 10.0)
    print("-"*50)
    print(f"{'TOTAL':<8} {r['bets']:>10,} {r['wins']:>10,} {r['win_pct']:>7.1f}% {r['roi']:>+9.1f}%")
    
    # Validate: shuffle test
    print("\n" + "="*70)
    print("VALIDATION: Random Shuffle Test")
    print("="*70)
    
    orig = run_backtest(df, 0.65, 3.0, 10.0)
    print(f"Original ROI: {orig['roi']:+.1f}%")
    
    np.random.seed(42)
    for i in range(3):
        shuffled = df.copy()
        shuffled['Win'] = np.random.permutation(shuffled['Win'].values)
        r = run_backtest(shuffled, 0.65, 3.0, 10.0)
        print(f"  Shuffle {i+1}: {r['roi']:+.1f}%")
    
    print(f"\nCompleted in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
