"""
Strategy Validation Suite
Tests to determine if +150% ROI is real or artifact.

Tests:
A. Out-of-Sample Test (Year-by-Year)
B. Track Breakdown (Profit by Track)
C. Data Age Audit (How old are the "last 5" splits?)
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

def run_validation_suite():
    conn = sqlite3.connect(DB_PATH)
    
    # =========================================================================
    # STEP 1: LOAD ALL DATA
    # =========================================================================
    progress("Loading race data...")
    races_query = f"""
    SELECT
        rm.MeetingID,
        rm.MeetingDate,
        t.TrackName,
        r.RaceNumber,
        r.Distance,
        ge.GreyhoundID,
        g.GreyhoundName,
        ge.Box,
        ge.StartingPrice as CurrentOdds,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '{start_date}'
      AND rm.MeetingDate <= '{end_date}'
      AND ge.StartingPrice IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.Box IS NOT NULL
    """
    races_df = pd.read_sql_query(races_query, conn)
    
    # Clean
    races_df['Box'] = pd.to_numeric(races_df['Box'], errors='coerce')
    races_df['CurrentOdds'] = pd.to_numeric(races_df['CurrentOdds'], errors='coerce')
    races_df['Distance'] = pd.to_numeric(races_df['Distance'], errors='coerce')
    races_df['MeetingDate'] = pd.to_datetime(races_df['MeetingDate'])
    races_df['RaceKey'] = races_df['MeetingID'].astype(str) + '_R' + races_df['RaceNumber'].astype(str)
    races_df['FieldSize'] = races_df.groupby('RaceKey')['GreyhoundID'].transform('count')
    
    progress(f"Loaded {len(races_df):,} entries")

    # =========================================================================
    # STEP 2: LOAD HISTORICAL DATA WITH DATES
    # =========================================================================
    progress("Loading full historical data with dates...")
    
    unique_dogs = races_df['GreyhoundID'].unique()
    dogs_str = ",".join(map(str, unique_dogs))
    
    history_query = f"""
    SELECT
        ge.GreyhoundID,
        rm.MeetingDate as HistDate,
        ge.Split,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney,
        (ge.FinishTimeBenchmarkLengths + COALESCE(rm.MeetingAvgBenchmarkLengths, 0)) as TotalPace
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.GreyhoundID IN ({dogs_str})
      AND rm.MeetingDate < '{end_date}'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate ASC
    """
    history_df = pd.read_sql_query(history_query, conn)
    history_df['HistDate'] = pd.to_datetime(history_df['HistDate'])
    
    progress(f"Loaded {len(history_df):,} historical records")

    # =========================================================================
    # STEP 3: CALCULATE ROLLING STATS + DATA AGE
    # For each dog, for each race, calculate:
    # - HistAvgSplit (last 5 recorded)
    # - HistAvgPace (last 5 recorded)
    # - OldestSplitAge (days since oldest of the 5 splits used)
    # - NewestSplitAge (days since newest of the 5 splits used)
    # =========================================================================
    progress("Computing rolling stats with age tracking...")
    
    # Split history only (where Split is not NaN)
    split_hist = history_df.dropna(subset=['Split']).copy()
    split_hist = split_hist.sort_values(['GreyhoundID', 'HistDate'])
    
    # For each dog, compute rolling mean of last 5 recorded splits
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['Split'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Track the age of the oldest split in the window
    # This is the date of the 5th-oldest split in the rolling window
    split_hist['SplitDate'] = split_hist['HistDate']
    split_hist['OldestSplitDate'] = split_hist.groupby('GreyhoundID')['SplitDate'].transform(
        lambda x: x.shift(5)
    )
    
    # Pace history
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist = pace_hist.sort_values(['GreyhoundID', 'HistDate'])
    pace_hist['HistAvgPace'] = pace_hist.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Prize money
    prize_hist = history_df.sort_values(['GreyhoundID', 'HistDate']).copy()
    prize_hist['CumPrize'] = prize_hist.groupby('GreyhoundID')['PrizeMoney'].cumsum()
    prize_hist['RunningPrize'] = prize_hist.groupby('GreyhoundID')['CumPrize'].shift(1).fillna(0)
    
    progress("Merging stats to race data...")
    
    # Merge - using HistDate as the date the stat was calculated for
    races_df = races_df.merge(
        split_hist.dropna(subset=['HistAvgSplit'])[['GreyhoundID', 'HistDate', 'HistAvgSplit', 'OldestSplitDate']],
        left_on=['GreyhoundID', 'MeetingDate'],
        right_on=['GreyhoundID', 'HistDate'],
        how='left'
    )
    
    races_df = races_df.merge(
        pace_hist.dropna(subset=['HistAvgPace'])[['GreyhoundID', 'HistDate', 'HistAvgPace']],
        left_on=['GreyhoundID', 'MeetingDate'],
        right_on=['GreyhoundID', 'HistDate'],
        how='left',
        suffixes=('', '_pace')
    )
    
    races_df = races_df.merge(
        prize_hist[['GreyhoundID', 'HistDate', 'RunningPrize']],
        left_on=['GreyhoundID', 'MeetingDate'],
        right_on=['GreyhoundID', 'HistDate'],
        how='left',
        suffixes=('', '_prize')
    )
    
    races_df['RunningPrize'] = races_df['RunningPrize'].fillna(0)
    
    # Calculate split age in days
    races_df['SplitAgeDays'] = (races_df['MeetingDate'] - races_df['OldestSplitDate']).dt.days
    
    # Market stats
    races_df['ImpliedProb'] = 1.0 / races_df['CurrentOdds'].replace(0, np.nan)
    market_stats = races_df.groupby('RaceKey').agg({
        'ImpliedProb': 'sum',
        'CurrentOdds': 'min'
    }).rename(columns={'ImpliedProb': 'MarketOverround', 'CurrentOdds': 'MinMarketOdds'}).reset_index()
    races_df = races_df.merge(market_stats, on='RaceKey', how='left')
    
    # =========================================================================
    # STEP 4: APPLY GUI FILTERS (including NZ exclusion)
    # =========================================================================
    filters = (
        (races_df['Distance'] <= 600) &
        (races_df['FieldSize'] >= 6) &
        (races_df['MarketOverround'] <= 1.40) &
        (~races_df['TrackName'].str.contains('(NZ)', regex=False, na=False)) & # Exclude NZ
        (~races_df['TrackName'].isin(['Hobart', 'Launceston', 'Devonport', 'Dport @ LCN'])) # Exclude TAS
    )
    df = races_df[filters].copy()
    df = df.dropna(subset=['HistAvgSplit', 'HistAvgPace'])
    
    progress(f"Qualified runners after filters: {len(df):,}")
    
    # Strategy logic
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
    df['PredictedPIR'] = df['HistAvgSplit'] + df['BoxAdj']
    
    df['PredictedPIRRank'] = df.groupby('RaceKey')['PredictedPIR'].rank(method='min')
    df['PaceRank'] = df.groupby('RaceKey')['HistAvgPace'].rank(method='min', ascending=True)
    
    df['IsPIRLeader'] = df['PredictedPIRRank'] == 1
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPaceTop3'] = df['PaceRank'] <= 3
    df['HasMoney'] = df['RunningPrize'] >= 30000
    df['InOddsRange'] = (df['CurrentOdds'] >= min_odds) & (df['CurrentOdds'] <= max_odds)
    
    def get_stake(odds):
        if odds < 3: return 0.5
        elif odds < 5: return 0.75
        elif odds < 10: return 1.0
        elif odds < 20: return 1.5
        else: return 2.0

    df['Stake'] = df['CurrentOdds'].apply(get_stake)
    df['Return'] = df.apply(lambda r: r['Stake'] * r['CurrentOdds'] if r['IsWinner'] else 0, axis=1)
    df['Profit'] = df['Return'] - df['Stake']
    df['Year'] = df['MeetingDate'].dt.year
    
    # Filter to strategy bets (Leader)
    leader_mask = df['IsPIRLeader'] & df['IsPaceLeader'] & df['HasMoney'] & df['InOddsRange']
    leader_bets = df[leader_mask].copy()
    
    progress(f"Leader strategy bets: {len(leader_bets):,}")
    
    # =========================================================================
    # TEST A: OUT-OF-SAMPLE (YEAR-BY-YEAR)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST A: OUT-OF-SAMPLE (YEAR-BY-YEAR)")
    print("="*80)
    print(f"{'Year':<6} | {'Bets':<8} | {'Win %':<8} | {'Profit':<12} | {'ROI':<8}")
    print("-" * 60)
    
    yearly_results = []
    for year in sorted(leader_bets['Year'].unique()):
        subset = leader_bets[leader_bets['Year'] == year]
        if len(subset) == 0:
            continue
        bets = len(subset)
        wins = subset['IsWinner'].sum()
        profit = subset['Profit'].sum()
        stake = subset['Stake'].sum()
        roi = (profit / stake * 100) if stake > 0 else 0
        win_rate = (wins / bets * 100) if bets > 0 else 0
        
        print(f"{year:<6} | {bets:<8} | {win_rate:<8.1f} | {profit:<12.2f} | {roi:<8.1f}%")
        yearly_results.append({'Year': year, 'Bets': bets, 'ROI': roi})
    
    # =========================================================================
    # TEST B: TRACK BREAKDOWN
    # =========================================================================
    print("\n" + "="*80)
    print("TEST B: TRACK BREAKDOWN (Top 20 by Bet Count)")
    print("="*80)
    print(f"{'Track':<25} | {'Bets':<6} | {'Win %':<6} | {'Profit':<10} | {'ROI':<8}")
    print("-" * 70)
    
    track_stats = leader_bets.groupby('TrackName').agg({
        'IsWinner': ['count', 'sum'],
        'Profit': 'sum',
        'Stake': 'sum'
    }).reset_index()
    
    track_stats.columns = ['Track', 'Bets', 'Wins', 'Profit', 'Stake']
    track_stats['WinRate'] = track_stats['Wins'] / track_stats['Bets'] * 100
    track_stats['ROI'] = track_stats['Profit'] / track_stats['Stake'] * 100
    track_stats = track_stats.sort_values('Bets', ascending=False).head(20)
    
    for _, row in track_stats.iterrows():
        track = row['Track'][:24]
        print(f"{track:<25} | {int(row['Bets']):<6} | {row['WinRate']:<6.1f} | {row['Profit']:<10.2f} | {row['ROI']:<8.1f}%")
    
    # Check concentration
    total_profit = leader_bets['Profit'].sum()
    top3_profit = track_stats.head(3)['Profit'].sum()
    top3_pct = (top3_profit / total_profit * 100) if total_profit > 0 else 0
    
    print("-" * 70)
    print(f"Top 3 tracks contribute {top3_pct:.1f}% of total profit")
    
    # Tracks with negative ROI
    negative_tracks = track_stats[track_stats['ROI'] < 0]
    print(f"Tracks with NEGATIVE ROI: {len(negative_tracks)} out of {len(track_stats)}")
    
    # =========================================================================
    # TEST C: DATA AGE AUDIT
    # =========================================================================
    print("\n" + "="*80)
    print("TEST C: DATA AGE AUDIT")
    print("How old is the OLDEST split in the 'last 5 recorded' window?")
    print("="*80)
    
    # Filter to only bets with valid age data
    age_bets = leader_bets.dropna(subset=['SplitAgeDays'])
    
    if len(age_bets) > 0:
        print(f"\nAnalyzing {len(age_bets):,} bets with tracked split age")
        
        # Stats
        print(f"\nSplit Age Statistics (days since oldest of 5 splits):")
        print(f"  Mean:   {age_bets['SplitAgeDays'].mean():.1f} days")
        print(f"  Median: {age_bets['SplitAgeDays'].median():.1f} days")
        print(f"  Min:    {age_bets['SplitAgeDays'].min():.1f} days")
        print(f"  Max:    {age_bets['SplitAgeDays'].max():.1f} days")
        
        # Buckets
        age_buckets = pd.cut(age_bets['SplitAgeDays'], 
                            bins=[0, 30, 60, 90, 180, 365, 9999],
                            labels=['<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>1yr'])
        
        bucket_stats = age_bets.groupby(age_buckets, observed=True).agg({
            'IsWinner': ['count', 'sum'],
            'Profit': 'sum',
            'Stake': 'sum'
        })
        bucket_stats.columns = ['Bets', 'Wins', 'Profit', 'Stake']
        bucket_stats['ROI'] = bucket_stats['Profit'] / bucket_stats['Stake'] * 100
        
        print(f"\nROI by Split Age Bucket:")
        print(f"{'Age Bucket':<12} | {'Bets':<8} | {'Profit':<12} | {'ROI':<8}")
        print("-" * 50)
        for bucket, row in bucket_stats.iterrows():
            print(f"{bucket:<12} | {int(row['Bets']):<8} | {row['Profit']:<12.2f} | {row['ROI']:<8.1f}%")
        
        # Check if old data performs better (red flag)
        fresh = age_bets[age_bets['SplitAgeDays'] <= 60]
        stale = age_bets[age_bets['SplitAgeDays'] > 180]
        
        if len(fresh) > 0 and len(stale) > 0:
            fresh_roi = (fresh['Profit'].sum() / fresh['Stake'].sum()) * 100
            stale_roi = (stale['Profit'].sum() / stale['Stake'].sum()) * 100
            
            print(f"\n⚠️  STALENESS CHECK:")
            print(f"  Fresh data (≤60 days old): {len(fresh):,} bets, ROI = {fresh_roi:.1f}%")
            print(f"  Stale data (>180 days old): {len(stale):,} bets, ROI = {stale_roi:.1f}%")
            
            if stale_roi > fresh_roi + 20:
                print(f"\n  🚨 RED FLAG: Stale data outperforms fresh by {stale_roi - fresh_roi:.1f}%!")
                print(f"     This suggests the strategy may be exploiting data artifacts.")
            else:
                print(f"\n  ✅ Fresh data performs comparably or better than stale. Good sign.")
    else:
        print("No split age data available for audit.")
    
    # =========================================================================
    # SAVE REPORT
    # =========================================================================
    with open('results/validation_report.txt', 'w') as f:
        f.write("STRATEGY VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*80 + "\n")
        f.write("Summary of findings printed to console.\n")
    
    conn.close()
    print("\n" + "="*80)
    print("Validation complete. Report saved to results/validation_report.txt")

if __name__ == "__main__":
    run_validation_suite()
