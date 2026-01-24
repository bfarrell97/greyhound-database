"""
Enhanced Betting Model v2 - FAST VERSION
Uses pre-computed aggregates instead of correlated subqueries
"""
import sqlite3
import pandas as pd
from datetime import datetime
import time

DB_PATH = "greyhound_racing.db"

def run_backtest():
    conn = sqlite3.connect(DB_PATH)
    
    print("=" * 80)
    print("ENHANCED BETTING MODEL v2 - FAST BACKTEST")
    print("=" * 80)
    start_time = time.time()
    
    # Step 1: Load base data (fast query, no subqueries)
    print("\n[1/5] Loading base race data...", end=" ", flush=True)
    t0 = time.time()
    
    base_query = """
    SELECT 
        ge.GreyhoundID,
        ge.RaceID,
        rm.MeetingDate,
        ge.Position,
        ge.StartingPrice,
        ge.Box,
        ge.JumpCode,
        ge.AverageSpeed,
        ge.CareerPrizeMoney,
        ge.FinishTimeBenchmarkLengths,
        rm.MeetingAvgBenchmarkLengths,
        g.DateWhelped,
        t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2022-01-01'
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.StartingPrice > 0
      AND ge.StartingPrice <= 20
    """
    
    df = pd.read_sql_query(base_query, conn)
    print(f"Done! {len(df):,} rows in {time.time()-t0:.1f}s")
    
    # Step 2: Convert and calculate basic fields
    print("[2/5] Processing basic fields...", end=" ", flush=True)
    t0 = time.time()
    
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['Won'] = (df['Position'] == 1).astype(int)
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate']).dt.tz_localize(None)
    df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce').dt.tz_localize(None)
    df['AgeYears'] = ((df['MeetingDate'] - df['DateWhelped']).dt.days / 365).fillna(2)
    df['TotalBenchmark'] = df['FinishTimeBenchmarkLengths'] + df['MeetingAvgBenchmarkLengths'].fillna(0)
    
    # Filter out rows with invalid prices
    df = df[df['StartingPrice'].notna() & (df['StartingPrice'] > 0)].copy()
    
    print(f"Done! {time.time()-t0:.1f}s")
    
    # Step 3: Calculate historical stats per dog (vectorized)
    print("[3/5] Computing historical pace & form (this takes ~30s)...", flush=True)
    t0 = time.time()
    
    # Sort by dog and date
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Calculate rolling stats using pandas groupby
    # Rolling win rate (last 5 races)
    df['RecentWins'] = df.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df['RecentRaces'] = df.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).count()
    ).fillna(0)
    df['RecentWinRate'] = (df['RecentWins'] / df['RecentRaces'].replace(0, 1)) * 100
    
    print(f"  - Win rate calculated in {time.time()-t0:.1f}s")
    
    # Rolling pace (last 5 races)
    df['HistoricalPace'] = df.groupby('GreyhoundID')['TotalBenchmark'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    print(f"  - Pace calculated, total: {time.time()-t0:.1f}s")
    
    # Step 4: Build scoring model
    print("[4/5] Building scoring model...", end=" ", flush=True)
    t0 = time.time()
    
    # Filter to dogs with enough history
    df = df[df['RecentRaces'] >= 3].copy()
    print(f"{len(df):,} rows with history...", end=" ", flush=True)
    
    # JumpScore
    df['JumpScore'] = df['JumpCode'].map({'Quick': 1.0, 'Medium': 0.5, 'Slow': 0.0}).fillna(0.3)
    
    # AgeScore
    df['AgeScore'] = df['AgeYears'].apply(lambda a: 1.0 if a <= 2 else (0.7 if a <= 3 else (0.4 if a <= 4 else 0.2)))
    
    # CareerScore
    def career_score(p):
        if pd.isna(p): return 0.3
        if p >= 50000: return 1.0
        if p >= 30000: return 0.8
        if p >= 15000: return 0.6
        if p >= 5000: return 0.4
        return 0.2
    df['CareerScore'] = df['CareerPrizeMoney'].apply(career_score)
    
    # SpeedRank (percentile within race)
    df['SpeedRank'] = df.groupby('RaceID')['AverageSpeed'].rank(pct=True, na_option='keep').fillna(0.5)
    
    # PaceScore (normalized)
    pace_min, pace_max = df['HistoricalPace'].quantile([0.05, 0.95])
    df['PaceScore'] = ((df['HistoricalPace'] - pace_min) / (pace_max - pace_min)).clip(0, 1).fillna(0.5)
    
    # FormScore
    df['FormScore'] = (df['RecentWinRate'] / 100).fillna(0)
    
    # BoxScore
    df['BoxScore'] = df['Box'].apply(lambda b: 0.8 if b in [1,2] else (0.5 if b in [3,4,5,6] else 0.4) if pd.notna(b) else 0.5)
    
    # Combined score
    df['TotalScore'] = (
        df['JumpScore'] * 0.25 +
        df['PaceScore'] * 0.20 +
        df['FormScore'] * 0.15 +
        df['CareerScore'] * 0.15 +
        df['AgeScore'] * 0.10 +
        df['SpeedRank'] * 0.10 +
        df['BoxScore'] * 0.05
    )
    
    print(f"Done! {time.time()-t0:.1f}s")
    
    # Step 5: Run backtests
    print("[5/5] Running backtests...", flush=True)
    t0 = time.time()
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    # All configurations to test
    configs = []
    for min_score in [0.5, 0.55, 0.6, 0.65, 0.7]:
        for price_min, price_max in [(1.5, 3.0), (1.5, 5.0), (2.0, 5.0), (3.0, 10.0)]:
            for jump_filter in [None, 'Quick']:
                configs.append((min_score, price_min, price_max, jump_filter))
    
    results = []
    for i, (min_score, price_min, price_max, jump_filter) in enumerate(configs):
        mask = (
            (df['TotalScore'] >= min_score) &
            (df['StartingPrice'] >= price_min) &
            (df['StartingPrice'] <= price_max)
        )
        if jump_filter:
            mask &= (df['JumpCode'] == jump_filter)
        
        subset = df[mask]
        if len(subset) < 30:
            continue
        
        wins = subset['Won'].sum()
        bets = len(subset)
        returns = subset[subset['Won'] == 1]['StartingPrice'].sum()
        roi = (returns - bets) / bets * 100
        
        results.append({
            'Score': min_score,
            'Price': f"${price_min:.1f}-${price_max:.1f}",
            'Jump': jump_filter or 'All',
            'Bets': bets,
            'Wins': wins,
            'Win%': wins/bets*100,
            'ROI': roi
        })
    
    results_df = pd.DataFrame(results).sort_values('ROI', ascending=False)
    
    print(f"\n{'Score':>6} {'Price':>12} {'Jump':>6} {'Bets':>8} {'Wins':>6} {'Win%':>7} {'ROI':>9}")
    print("-" * 60)
    for _, r in results_df.head(25).iterrows():
        print(f"{r['Score']:>6.2f} {r['Price']:>12} {r['Jump']:>6} {r['Bets']:>8,} {r['Wins']:>6,} {r['Win%']:>6.1f}% {r['ROI']:>+8.1f}%")
    
    # Best model details
    print("\n" + "=" * 80)
    print("BEST MODEL: Quick + Score >= 0.6 + $1.50-$5.00")
    print("=" * 80)
    
    best = df[
        (df['TotalScore'] >= 0.6) &
        (df['JumpCode'] == 'Quick') &
        (df['StartingPrice'] >= 1.5) &
        (df['StartingPrice'] <= 5.0)
    ].copy()
    
    wins = best['Won'].sum()
    bets = len(best)
    returns = best[best['Won'] == 1]['StartingPrice'].sum()
    roi = (returns - bets) / bets * 100
    
    print(f"\nTotal bets: {bets:,}")
    print(f"Wins: {wins:,} ({wins/bets*100:.1f}%)")
    print(f"ROI: {roi:+.1f}%")
    
    # Monthly breakdown
    print("\n--- Monthly Breakdown ---")
    best['Month'] = best['MeetingDate'].dt.to_period('M')
    monthly = best.groupby('Month').agg(
        Bets=('Won', 'count'),
        Wins=('Won', 'sum'),
        Returns=('StartingPrice', lambda x: x[best.loc[x.index, 'Won'] == 1].sum())
    ).reset_index()
    monthly['ROI'] = (monthly['Returns'] - monthly['Bets']) / monthly['Bets'] * 100
    
    print(f"{'Month':>10} {'Bets':>6} {'Wins':>5} {'Win%':>7} {'ROI':>9}")
    print("-" * 42)
    for _, row in monthly.tail(12).iterrows():
        print(f"{str(row['Month']):>10} {row['Bets']:>6} {row['Wins']:>5} {row['Wins']/row['Bets']*100:>6.1f}% {row['ROI']:>+8.1f}%")
    
    # Daily volume
    best['Date'] = best['MeetingDate'].dt.date
    daily = best.groupby('Date').size()
    print(f"\n--- Daily Volume ---")
    print(f"Avg bets/day: {daily.mean():.1f}")
    print(f"Median: {daily.median():.0f}")
    print(f"Days with 3-10 bets: {((daily >= 3) & (daily <= 10)).sum()}/{len(daily)}")
    
    # Premium filter
    print("\n" + "=" * 80)
    print("PREMIUM: Quick + Age<=2 + Career $15k+")
    print("=" * 80)
    
    for price_min, price_max in [(1.5, 2.0), (1.5, 3.0), (1.5, 5.0), (2.0, 5.0)]:
        prem = df[
            (df['JumpCode'] == 'Quick') &
            (df['AgeYears'] <= 2) &
            (df['CareerPrizeMoney'] >= 15000) &
            (df['StartingPrice'] >= price_min) &
            (df['StartingPrice'] <= price_max)
        ]
        if len(prem) < 20:
            continue
        wins = prem['Won'].sum()
        bets = len(prem)
        returns = prem[prem['Won'] == 1]['StartingPrice'].sum()
        roi = (returns - bets) / bets * 100
        print(f"${price_min:.1f}-${price_max:.1f}: {bets:>6,} bets, {wins:>5,} wins ({wins/bets*100:>5.1f}%), ROI: {roi:>+7.1f}%")
    
    conn.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED in {total_time:.1f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_backtest()
