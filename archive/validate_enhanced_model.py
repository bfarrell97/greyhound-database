"""
COMPREHENSIVE MODEL VALIDATION
Tests for overfitting, consistency, and robustness
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time

DB_PATH = 'greyhound_racing.db'

def load_data():
    """Load all historical data with enhanced fields"""
    print("Loading data...")
    start = time.time()
    
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.GreyhoundID, ge.RaceID, ge.Position, ge.Box,
        ge.StartingPrice, ge.FinishTimeBenchmarkLengths, ge.JumpCode,
        ge.CareerPrizeMoney, ge.AverageSpeed,
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
    ORDER BY rm.MeetingDate, r.RaceID, ge.Position
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    return df

def prepare_features(df):
    """Prepare all features for scoring"""
    print("Preparing features...")
    start = time.time()
    
    # Convert types
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
    df['AverageSpeed'] = pd.to_numeric(df['AverageSpeed'], errors='coerce')
    df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')
    df['MeetingAvgBenchmarkLengths'] = pd.to_numeric(df['MeetingAvgBenchmarkLengths'], errors='coerce')
    
    # Win flag
    df['Win'] = (df['Position'] == '1') | (df['Position'] == 1)
    
    # Age calculation
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce', utc=True).dt.tz_localize(None)
    df['AgeYears'] = (df['MeetingDate'] - df['DateWhelped']).dt.days / 365.25
    
    # Sort for historical calculations
    df = df.sort_values(['MeetingDate', 'RaceID']).reset_index(drop=True)
    
    print(f"  Done in {time.time()-start:.1f}s")
    return df

def calculate_historical_stats(df):
    """Calculate expanding historical stats for each dog"""
    print("Calculating historical stats (this takes ~60s)...")
    start = time.time()
    
    # Win rate - expanding mean up to but not including current race
    df['CumWins'] = df.groupby('GreyhoundID')['Win'].cumsum().shift(1).fillna(0)
    df['CumRaces'] = df.groupby('GreyhoundID').cumcount()
    df['WinRate'] = np.where(df['CumRaces'] > 0, df['CumWins'] / df['CumRaces'], 0)
    
    # Pace - expanding mean
    df['TotalBench'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths']
    df['CumPace'] = df.groupby('GreyhoundID')['TotalBench'].cumsum().shift(1).fillna(0)
    df['AvgPace'] = np.where(df['CumRaces'] > 0, df['CumPace'] / df['CumRaces'], 0)
    
    # Require minimum 3 races
    df['HasHistory'] = df['CumRaces'] >= 3
    
    print(f"  Done in {time.time()-start:.1f}s")
    print(f"  Rows with 3+ races history: {df['HasHistory'].sum():,}")
    return df

def calculate_scores(df):
    """Calculate composite score for betting"""
    print("Calculating scores...")
    
    scored = df[df['HasHistory']].copy()
    
    # JumpCode score
    scored['JumpScore'] = scored['JumpCode'].map({'Quick': 3, 'Average': 1, 'Slow': 0}).fillna(1)
    
    # Age score (younger = better)
    scored['AgeScore'] = np.where(scored['AgeYears'] <= 2, 2,
                         np.where(scored['AgeYears'] <= 3, 1, 0))
    
    # Career money score
    scored['MoneyScore'] = np.where(scored['CareerPrizeMoney'] >= 50000, 2,
                           np.where(scored['CareerPrizeMoney'] >= 15000, 1.5,
                           np.where(scored['CareerPrizeMoney'] >= 5000, 1, 0)))
    
    # Speed quintile (per race)
    scored['SpeedQuintile'] = scored.groupby('RaceID')['AverageSpeed'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=[0,1,2,3,4], duplicates='drop') 
        if len(x.dropna()) >= 5 else 2
    )
    scored['SpeedQuintile'] = pd.to_numeric(scored['SpeedQuintile'], errors='coerce').fillna(2)
    scored['SpeedScore'] = scored['SpeedQuintile'] / 2  # 0-2 scale
    
    # Win rate score (0-2 scale)
    scored['WinRateScore'] = scored['WinRate'] * 10  # Max ~2 for 20% win rate
    scored['WinRateScore'] = scored['WinRateScore'].clip(0, 2)
    
    # Pace score (higher = better, normalize per race)
    scored['PaceRank'] = scored.groupby('RaceID')['AvgPace'].rank(ascending=False, method='min')
    scored['PaceScore'] = np.where(scored['PaceRank'] <= 2, 2,
                          np.where(scored['PaceRank'] <= 4, 1, 0))
    
    # Combined score (0-1 scale)
    max_score = 3 + 2 + 2 + 2 + 2 + 2  # 13 max
    scored['RawScore'] = (scored['JumpScore'] + scored['AgeScore'] + scored['MoneyScore'] + 
                          scored['SpeedScore'] + scored['WinRateScore'] + scored['PaceScore'])
    scored['Score'] = scored['RawScore'] / max_score
    
    print(f"  Scored {len(scored):,} rows")
    return scored

def run_backtest(df, min_score, price_min, price_max, quick_only=False, year=None):
    """Run backtest with given parameters"""
    subset = df.copy()
    
    if year:
        subset = subset[subset['MeetingDate'].dt.year == year]
    
    subset = subset[
        (subset['Score'] >= min_score) &
        (subset['StartingPrice'] >= price_min) &
        (subset['StartingPrice'] <= price_max)
    ]
    
    if quick_only:
        subset = subset[subset['JumpCode'] == 'Quick']
    
    if len(subset) == 0:
        return {'bets': 0, 'wins': 0, 'win_pct': 0, 'roi': 0, 'profit': 0}
    
    bets = len(subset)
    wins = subset['Win'].sum()
    returns = (subset[subset['Win']]['StartingPrice'] - 1).sum()
    profit = returns - bets
    
    return {
        'bets': bets,
        'wins': wins,
        'win_pct': wins / bets * 100,
        'roi': profit / bets * 100,
        'profit': profit
    }

def test_year_by_year(df):
    """Test model on each year separately"""
    print("\n" + "="*70)
    print("TEST 1: YEAR-BY-YEAR CONSISTENCY")
    print("="*70)
    print("Testing if model works across all years (not just 2025)\n")
    
    years = sorted(df['MeetingDate'].dt.year.unique())
    
    print(f"{'Year':<8} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'ROI':>10} {'Profit':>10}")
    print("-" * 60)
    
    total_bets = 0
    total_profit = 0
    
    for year in years:
        result = run_backtest(df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True, year=year)
        if result['bets'] > 0:
            print(f"{year:<8} {result['bets']:>8,} {result['wins']:>8,} {result['win_pct']:>7.1f}% {result['roi']:>+9.1f}% {result['profit']:>+10.1f}")
            total_bets += result['bets']
            total_profit += result['profit']
    
    print("-" * 60)
    if total_bets > 0:
        print(f"{'TOTAL':<8} {total_bets:>8,} {'-':>8} {'-':>8} {total_profit/total_bets*100:>+9.1f}% {total_profit:>+10.1f}")
    
    return total_profit / total_bets * 100 if total_bets > 0 else 0

def test_train_test_split(df):
    """Train on 2020-2023, test on 2024-2025"""
    print("\n" + "="*70)
    print("TEST 2: TRAIN/TEST SPLIT (2020-2023 vs 2024-2025)")
    print("="*70)
    print("Model uses same weights on both - testing for overfitting\n")
    
    train = df[df['MeetingDate'].dt.year <= 2023]
    test = df[df['MeetingDate'].dt.year >= 2024]
    
    print("TRAINING SET (2020-2023):")
    train_result = run_backtest(train, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
    print(f"  Bets: {train_result['bets']:,}, Wins: {train_result['wins']:,} ({train_result['win_pct']:.1f}%), ROI: {train_result['roi']:+.1f}%")
    
    print("\nTEST SET (2024-2025):")
    test_result = run_backtest(test, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
    print(f"  Bets: {test_result['bets']:,}, Wins: {test_result['wins']:,} ({test_result['win_pct']:.1f}%), ROI: {test_result['roi']:+.1f}%")
    
    diff = abs(train_result['roi'] - test_result['roi'])
    if diff < 15:
        print(f"\n✓ PASSED: Train/Test ROI difference is only {diff:.1f}% - model is robust")
    else:
        print(f"\n⚠ WARNING: Train/Test ROI difference is {diff:.1f}% - possible overfitting")
    
    return train_result['roi'], test_result['roi']

def test_random_shuffles(df):
    """Test if results hold under random date shuffling (should fail if model is real)"""
    print("\n" + "="*70)
    print("TEST 3: RANDOM SHUFFLE TEST")
    print("="*70)
    print("If we shuffle dates randomly, ROI should drop to near-zero\n")
    
    # Original result
    orig = run_backtest(df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
    print(f"Original ROI: {orig['roi']:+.1f}%")
    
    # Shuffle test
    np.random.seed(42)
    shuffle_rois = []
    
    for i in range(5):
        shuffled = df.copy()
        # Shuffle win outcomes while keeping everything else fixed
        shuffled['Win'] = np.random.permutation(shuffled['Win'].values)
        result = run_backtest(shuffled, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
        shuffle_rois.append(result['roi'])
        print(f"  Shuffle {i+1}: {result['roi']:+.1f}%")
    
    avg_shuffle = np.mean(shuffle_rois)
    print(f"\nAverage shuffled ROI: {avg_shuffle:+.1f}%")
    
    if orig['roi'] > avg_shuffle + 20:
        print(f"✓ PASSED: Original ROI ({orig['roi']:+.1f}%) is significantly better than random ({avg_shuffle:+.1f}%)")
    else:
        print(f"⚠ WARNING: Original ROI not much better than random shuffles")
    
    return orig['roi'], avg_shuffle

def test_by_track(df):
    """Test performance across different tracks"""
    print("\n" + "="*70)
    print("TEST 4: TRACK-BY-TRACK CONSISTENCY")
    print("="*70)
    print("Testing if model works across different tracks\n")
    
    track_results = []
    tracks = df['TrackName'].value_counts().head(15).index
    
    print(f"{'Track':<25} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 65)
    
    for track in tracks:
        track_df = df[df['TrackName'] == track]
        result = run_backtest(track_df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
        if result['bets'] >= 20:
            print(f"{track[:25]:<25} {result['bets']:>8,} {result['wins']:>8,} {result['win_pct']:>7.1f}% {result['roi']:>+9.1f}%")
            track_results.append(result['roi'])
    
    profitable_tracks = sum(1 for r in track_results if r > 0)
    print(f"\n{profitable_tracks}/{len(track_results)} tracks are profitable ({profitable_tracks/len(track_results)*100:.0f}%)")
    
    return profitable_tracks / len(track_results) * 100

def test_by_distance(df):
    """Test performance across different distances"""
    print("\n" + "="*70)
    print("TEST 5: DISTANCE CONSISTENCY")
    print("="*70)
    print("Testing if model works across different race distances\n")
    
    df['DistanceBucket'] = pd.cut(df['Distance'], bins=[0, 400, 500, 600, 800, 2000], 
                                   labels=['<400m', '400-500m', '500-600m', '600-800m', '>800m'])
    
    print(f"{'Distance':<15} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 55)
    
    for dist in df['DistanceBucket'].dropna().unique():
        dist_df = df[df['DistanceBucket'] == dist]
        result = run_backtest(dist_df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
        if result['bets'] >= 20:
            print(f"{dist:<15} {result['bets']:>8,} {result['wins']:>8,} {result['win_pct']:>7.1f}% {result['roi']:>+9.1f}%")

def test_monthly_variance(df):
    """Test month-to-month variance"""
    print("\n" + "="*70)
    print("TEST 6: MONTHLY VARIANCE")
    print("="*70)
    print("Testing monthly ROI stability\n")
    
    df['YearMonth'] = df['MeetingDate'].dt.to_period('M')
    months = df['YearMonth'].unique()
    
    monthly_rois = []
    for month in sorted(months):
        month_df = df[df['YearMonth'] == month]
        result = run_backtest(month_df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
        if result['bets'] >= 10:
            monthly_rois.append(result['roi'])
    
    positive_months = sum(1 for r in monthly_rois if r > 0)
    print(f"Positive ROI months: {positive_months}/{len(monthly_rois)} ({positive_months/len(monthly_rois)*100:.0f}%)")
    print(f"Average monthly ROI: {np.mean(monthly_rois):+.1f}%")
    print(f"Std dev: {np.std(monthly_rois):.1f}%")
    print(f"Worst month: {min(monthly_rois):+.1f}%")
    print(f"Best month: {max(monthly_rois):+.1f}%")
    
    return positive_months / len(monthly_rois) * 100, np.mean(monthly_rois)

def test_score_thresholds(df):
    """Test different score thresholds"""
    print("\n" + "="*70)
    print("TEST 7: SCORE THRESHOLD SENSITIVITY")
    print("="*70)
    print("Testing if higher scores = better performance (as expected)\n")
    
    print(f"{'Score >=':<10} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 50)
    
    prev_roi = None
    monotonic = True
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        result = run_backtest(df, min_score=threshold, price_min=1.5, price_max=5.0, quick_only=True)
        if result['bets'] >= 20:
            print(f"{threshold:<10.2f} {result['bets']:>8,} {result['wins']:>8,} {result['win_pct']:>7.1f}% {result['roi']:>+9.1f}%")
            if prev_roi is not None and result['roi'] < prev_roi - 5:
                monotonic = False
            prev_roi = result['roi']
    
    if monotonic:
        print("\n✓ PASSED: Higher thresholds generally = better ROI (model is sensible)")
    else:
        print("\n⚠ Note: Non-monotonic relationship (normal due to sample size)")

def test_without_jumpcode(df):
    """Test how much JumpCode contributes"""
    print("\n" + "="*70)
    print("TEST 8: JUMPCODE CONTRIBUTION")
    print("="*70)
    print("Testing how much JumpCode='Quick' filter contributes\n")
    
    with_jump = run_backtest(df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=True)
    without_jump = run_backtest(df, min_score=0.60, price_min=1.5, price_max=5.0, quick_only=False)
    
    print(f"With JumpCode='Quick':    {with_jump['bets']:,} bets, {with_jump['win_pct']:.1f}% wins, {with_jump['roi']:+.1f}% ROI")
    print(f"Without filter:           {without_jump['bets']:,} bets, {without_jump['win_pct']:.1f}% wins, {without_jump['roi']:+.1f}% ROI")
    print(f"\nJumpCode adds: {with_jump['roi'] - without_jump['roi']:+.1f}% ROI, {with_jump['win_pct'] - without_jump['win_pct']:+.1f}% win rate")

def test_odds_bands(df):
    """Test across different odds bands"""
    print("\n" + "="*70)
    print("TEST 9: ODDS BAND ANALYSIS")
    print("="*70)
    print("Testing performance in different price ranges\n")
    
    bands = [
        (1.01, 1.50, "Short (<$1.50)"),
        (1.50, 2.00, "$1.50-$2.00"),
        (2.00, 3.00, "$2.00-$3.00"),
        (3.00, 5.00, "$3.00-$5.00"),
        (5.00, 10.00, "$5.00-$10.00"),
        (10.00, 50.00, "$10.00+"),
    ]
    
    print(f"{'Price Band':<20} {'Bets':>8} {'Wins':>8} {'Win%':>8} {'ROI':>10}")
    print("-" * 60)
    
    for pmin, pmax, label in bands:
        result = run_backtest(df, min_score=0.60, price_min=pmin, price_max=pmax, quick_only=True)
        if result['bets'] >= 20:
            print(f"{label:<20} {result['bets']:>8,} {result['wins']:>8,} {result['win_pct']:>7.1f}% {result['roi']:>+9.1f}%")

def main():
    print("="*70)
    print("ENHANCED MODEL VALIDATION SUITE")
    print("="*70)
    start_time = time.time()
    
    # Load and prepare data
    df = load_data()
    df = prepare_features(df)
    df = calculate_historical_stats(df)
    df = calculate_scores(df)
    
    # Run all tests
    year_roi = test_year_by_year(df)
    train_roi, test_roi = test_train_test_split(df)
    orig_roi, shuffle_roi = test_random_shuffles(df)
    track_pct = test_by_track(df)
    test_by_distance(df)
    month_pct, month_avg = test_monthly_variance(df)
    test_score_thresholds(df)
    test_without_jumpcode(df)
    test_odds_bands(df)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    tests_passed = 0
    total_tests = 6
    
    # Year consistency
    if year_roi > 20:
        print("✓ Year-by-year: Positive overall ROI across years")
        tests_passed += 1
    else:
        print("✗ Year-by-year: Poor performance across years")
    
    # Train/test split
    if abs(train_roi - test_roi) < 20:
        print("✓ Train/Test split: Consistent performance (no overfitting)")
        tests_passed += 1
    else:
        print("✗ Train/Test split: Large difference suggests overfitting")
    
    # Random shuffle
    if orig_roi > shuffle_roi + 15:
        print("✓ Random shuffle: Model beats random significantly")
        tests_passed += 1
    else:
        print("✗ Random shuffle: Model not much better than random")
    
    # Track consistency
    if track_pct >= 60:
        print(f"✓ Track consistency: {track_pct:.0f}% of tracks profitable")
        tests_passed += 1
    else:
        print(f"✗ Track consistency: Only {track_pct:.0f}% of tracks profitable")
    
    # Monthly consistency
    if month_pct >= 60 and month_avg > 20:
        print(f"✓ Monthly consistency: {month_pct:.0f}% months profitable, avg {month_avg:+.1f}%")
        tests_passed += 1
    else:
        print(f"✗ Monthly consistency: Only {month_pct:.0f}% months profitable")
    
    # Overall ROI
    if year_roi > 30:
        print(f"✓ Overall ROI: {year_roi:+.1f}% is strong")
        tests_passed += 1
    else:
        print(f"✗ Overall ROI: {year_roi:+.1f}% is weak")
    
    print(f"\n{'='*70}")
    print(f"RESULT: {tests_passed}/{total_tests} tests passed")
    if tests_passed >= 5:
        print("MODEL IS VALIDATED ✓")
    elif tests_passed >= 3:
        print("MODEL IS PARTIALLY VALIDATED - USE WITH CAUTION")
    else:
        print("MODEL FAILED VALIDATION - DO NOT USE")
    print(f"{'='*70}")
    print(f"\nCompleted in {time.time()-start_time:.1f} seconds")

if __name__ == "__main__":
    main()
