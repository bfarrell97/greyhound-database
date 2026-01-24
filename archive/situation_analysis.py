"""
SITUATION ANALYSIS - Find specific winning scenarios
Instead of predicting winners, find SITUATIONS where underdogs outperform
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_data():
    conn = sqlite3.connect(DB_PATH)
    
    # Get all race data with detailed features
    query = """
    SELECT 
        ge.GreyhoundID,
        ge.RaceID,
        ge.Position,
        ge.Box,
        ge.StartingPrice,
        ge.FinishTimeBenchmarkLengths as G_OT,
        ge.SplitBenchmarkLengths as G_Split,
        r.Distance,
        r.RaceNumber,
        rm.MeetingDate,
        rm.MeetingAvgBenchmarkLengths as M_OT,
        rm.MeetingSplitAvgBenchmarkLengths as M_Split,
        t.TrackName,
        g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
        AND ge.Position IS NOT NULL
        AND ge.StartingPrice IS NOT NULL
        AND ge.StartingPrice > 0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_track_tier(track):
    metro = ['Wentworth Park', 'The Meadows', 'Albion Park', 'Sandown Park', 'Cannington']
    provincial = ['Richmond', 'Bulli', 'Shepparton', 'Ipswich', 'Dapto', 'Gosford', 'Warragul']
    if track in metro:
        return 'Metro'
    elif track in provincial:
        return 'Provincial'
    else:
        return 'Country'

def analyze_box_performance(df):
    """Analyze if certain boxes outperform at value odds"""
    log("\n" + "="*80)
    log("BOX POSITION ANALYSIS")
    log("="*80)
    
    # Filter to value range ($4-$15)
    value_df = df[(df['StartingPrice'] >= 4) & (df['StartingPrice'] <= 15)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    box_stats = value_df.groupby('Box').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    }).round(3)
    
    box_stats.columns = ['Wins', 'Bets', 'Strike%', 'TotalProfit', 'AvgOdds']
    box_stats['ROI%'] = (box_stats['TotalProfit'] / box_stats['Bets'] * 100).round(1)
    
    log(f"\nBox performance (dogs at $4-$15):")
    log("-" * 60)
    for box in sorted(box_stats.index):
        if box_stats.loc[box, 'Bets'] >= 1000:
            log(f"Box {box}: {box_stats.loc[box, 'Bets']:,} bets | {box_stats.loc[box, 'Strike%']*100:.1f}% | ROI {box_stats.loc[box, 'ROI%']:+.1f}%")

def analyze_race_number(df):
    """Earlier/later races may have different dynamics"""
    log("\n" + "="*80)
    log("RACE NUMBER ANALYSIS")
    log("="*80)
    
    value_df = df[(df['StartingPrice'] >= 3) & (df['StartingPrice'] <= 10)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    race_stats = value_df.groupby('RaceNumber').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    }).round(3)
    
    race_stats.columns = ['Wins', 'Bets', 'Strike%', 'TotalProfit', 'AvgOdds']
    race_stats['ROI%'] = (race_stats['TotalProfit'] / race_stats['Bets'] * 100).round(1)
    
    log(f"\nRace # performance (dogs at $3-$10):")
    log("-" * 60)
    for race_num in sorted(race_stats.index):
        if race_num <= 12 and race_stats.loc[race_num, 'Bets'] >= 1000:
            log(f"Race {race_num:2d}: {race_stats.loc[race_num, 'Bets']:,} bets | {race_stats.loc[race_num, 'Strike%']*100:.1f}% | ROI {race_stats.loc[race_num, 'ROI%']:+.1f}%")

def analyze_distance(df):
    """Different distances may have different value patterns"""
    log("\n" + "="*80)
    log("DISTANCE ANALYSIS")
    log("="*80)
    
    value_df = df[(df['StartingPrice'] >= 3) & (df['StartingPrice'] <= 10)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    # Common distances
    common_dists = value_df['Distance'].value_counts().head(15).index
    dist_df = value_df[value_df['Distance'].isin(common_dists)]
    
    dist_stats = dist_df.groupby('Distance').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    }).round(3)
    
    dist_stats.columns = ['Wins', 'Bets', 'Strike%', 'TotalProfit', 'AvgOdds']
    dist_stats['ROI%'] = (dist_stats['TotalProfit'] / dist_stats['Bets'] * 100).round(1)
    dist_stats = dist_stats.sort_values('ROI%', ascending=False)
    
    log(f"\nDistance performance (dogs at $3-$10, top 10):")
    log("-" * 60)
    for dist in dist_stats.head(10).index:
        if dist_stats.loc[dist, 'Bets'] >= 500:
            log(f"{dist}m: {dist_stats.loc[dist, 'Bets']:,} bets | {dist_stats.loc[dist, 'Strike%']*100:.1f}% | ROI {dist_stats.loc[dist, 'ROI%']:+.1f}%")

def analyze_market_conditions(df):
    """Analyze based on favourite strength"""
    log("\n" + "="*80)
    log("MARKET CONDITION ANALYSIS")
    log("="*80)
    
    # Get favourite price per race
    fav_prices = df.groupby('RaceID')['StartingPrice'].min().reset_index()
    fav_prices.columns = ['RaceID', 'FavPrice']
    df = df.merge(fav_prices, on='RaceID')
    
    # Categorize market
    df['MarketType'] = pd.cut(df['FavPrice'], 
                              bins=[0, 1.5, 2.0, 2.5, 3.0, 100],
                              labels=['HotFav<1.5', 'StrongFav1.5-2', 'ModFav2-2.5', 'WeakFav2.5-3', 'OpenRace3+'])
    
    # Look at second/third favourites in different market conditions
    value_df = df[(df['StartingPrice'] >= 3) & (df['StartingPrice'] <= 8)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    market_stats = value_df.groupby('MarketType').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    }).round(3)
    
    market_stats.columns = ['Wins', 'Bets', 'Strike%', 'TotalProfit', 'AvgOdds']
    market_stats['ROI%'] = (market_stats['TotalProfit'] / market_stats['Bets'] * 100).round(1)
    
    log(f"\nSecond choices ($3-$8) when favourite is:")
    log("-" * 60)
    for market in market_stats.index:
        if market_stats.loc[market, 'Bets'] >= 1000:
            log(f"{market:15s}: {market_stats.loc[market, 'Bets']:,} bets | {market_stats.loc[market, 'Strike%']*100:.1f}% | ROI {market_stats.loc[market, 'ROI%']:+.1f}%")

def analyze_form_patterns(df):
    """Analyze dogs with specific form patterns"""
    log("\n" + "="*80)
    log("FORM PATTERN ANALYSIS (using benchmark data)")
    log("="*80)
    
    # Calculate GM_OT for each entry
    df['GM_OT'] = df['G_OT'] - df['M_OT']
    
    # Get historical form for each dog
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Calculate rolling stats
    df['PrevGM_OT'] = df.groupby('GreyhoundID')['GM_OT'].shift(1)
    df['PrevPosition'] = df.groupby('GreyhoundID')['Position'].shift(1)
    
    # Look for "bouncebacks" - dogs that ran poorly but have good benchmarks
    log("\nBounceback pattern (last run poor position but good benchmark):")
    log("-" * 60)
    
    value_df = df[(df['StartingPrice'] >= 4) & (df['StartingPrice'] <= 12)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    # Pattern: Last run outside top 3 but had top 3 GM_OT in their last 5
    bouncebacks = value_df[(value_df['PrevPosition'] >= 4) & (value_df['PrevGM_OT'] > 0)]
    
    if len(bouncebacks) > 0:
        wins = bouncebacks['Won'].sum()
        bets = len(bouncebacks)
        strike = wins / bets * 100
        profit = bouncebacks['Profit'].sum()
        roi = profit / bets * 100
        log(f"Last run 4th+ but positive benchmark: {bets:,} bets | {strike:.1f}% | ROI {roi:+.1f}%")
    
    # Pattern: Coming off a win vs coming off a loss
    log("\nLast run result:")
    log("-" * 60)
    
    for prev_pos in [1, 2, 3, 4, 5, 6, 7, 8]:
        subset = value_df[value_df['PrevPosition'] == prev_pos]
        if len(subset) >= 500:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins / bets * 100
            profit = subset['Profit'].sum()
            roi = profit / bets * 100
            avg_odds = subset['StartingPrice'].mean()
            log(f"Last run {prev_pos}th: {bets:,} bets | {strike:.1f}% | ${avg_odds:.2f} avg | ROI {roi:+.1f}%")

def analyze_track_tier_value(df):
    """Analyze if track tier affects value"""
    log("\n" + "="*80)
    log("TRACK TIER VALUE ANALYSIS")
    log("="*80)
    
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    
    # Value range dogs
    value_df = df[(df['StartingPrice'] >= 3) & (df['StartingPrice'] <= 10)].copy()
    value_df['Won'] = value_df['Position'] == 1
    value_df['Profit'] = value_df.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    tier_stats = value_df.groupby('Tier').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    }).round(3)
    
    tier_stats.columns = ['Wins', 'Bets', 'Strike%', 'TotalProfit', 'AvgOdds']
    tier_stats['ROI%'] = (tier_stats['TotalProfit'] / tier_stats['Bets'] * 100).round(1)
    
    log(f"\nTrack tier performance ($3-$10 runners):")
    log("-" * 60)
    for tier in tier_stats.index:
        log(f"{tier:12s}: {tier_stats.loc[tier, 'Bets']:,} bets | {tier_stats.loc[tier, 'Strike%']*100:.1f}% | ROI {tier_stats.loc[tier, 'ROI%']:+.1f}%")

def analyze_multivariate_combinations(df):
    """Combine multiple factors to find winning patterns"""
    log("\n" + "="*80)
    log("MULTI-FACTOR COMBINATIONS (searching for edge)")
    log("="*80)
    
    # Get favourite price per race
    fav_prices = df.groupby('RaceID')['StartingPrice'].min().reset_index()
    fav_prices.columns = ['RaceID', 'FavPrice']
    df = df.merge(fav_prices, on='RaceID', how='left')
    
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    df['GM_OT'] = df['G_OT'] - df['M_OT']
    
    # Rolling metrics
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['PrevGM_OT'] = df.groupby('GreyhoundID')['GM_OT'].shift(1)
    df['PrevPosition'] = df.groupby('GreyhoundID')['Position'].shift(1)
    df['PrevWon'] = df['PrevPosition'] == 1
    
    # Filter to 2025 for consistency
    df_2025 = df[df['MeetingDate'] >= '2025-01-01'].copy()
    
    log("\nTesting factor combinations...")
    log("-" * 80)
    
    results = []
    
    # Test various combinations
    for tier in ['Country', 'Provincial', 'Metro', 'All']:
        for min_price, max_price in [(3, 6), (4, 8), (5, 10), (6, 12)]:
            for box in [1, 2, 8, 'All']:
                for prev_won in [True, False, 'All']:
                    for fav_range in ['weak', 'all']:
                        
                        # Apply filters
                        subset = df_2025.copy()
                        
                        if tier != 'All':
                            subset = subset[subset['Tier'] == tier]
                        
                        subset = subset[(subset['StartingPrice'] >= min_price) & 
                                       (subset['StartingPrice'] <= max_price)]
                        
                        if box != 'All':
                            subset = subset[subset['Box'] == box]
                        
                        if prev_won != 'All':
                            subset = subset[subset['PrevWon'] == prev_won]
                        
                        if fav_range == 'weak':
                            subset = subset[subset['FavPrice'] >= 2.5]
                        
                        if len(subset) < 100:
                            continue
                        
                        # Calculate results
                        subset['Won'] = subset['Position'] == 1
                        subset['Profit'] = subset.apply(
                            lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
                        )
                        
                        wins = subset['Won'].sum()
                        bets = len(subset)
                        strike = wins / bets * 100
                        profit = subset['Profit'].sum()
                        roi = profit / bets * 100
                        avg_odds = subset['StartingPrice'].mean()
                        
                        # Calculate days in period
                        days = (pd.to_datetime(subset['MeetingDate'].max()) - 
                               pd.to_datetime(subset['MeetingDate'].min())).days + 1
                        bets_per_day = bets / days if days > 0 else 0
                        
                        results.append({
                            'Tier': tier,
                            'Price': f'${min_price}-${max_price}',
                            'Box': box,
                            'PrevWon': prev_won,
                            'FavRange': fav_range,
                            'Bets': bets,
                            'BetsPerDay': bets_per_day,
                            'Strike%': strike,
                            'AvgOdds': avg_odds,
                            'Profit': profit,
                            'ROI%': roi
                        })
    
    results_df = pd.DataFrame(results)
    
    # Filter to meaningful sample sizes and positive ROI
    positive = results_df[(results_df['ROI%'] > 0) & (results_df['Bets'] >= 100)]
    
    if len(positive) > 0:
        log("\nPOSITIVE ROI configurations found:")
        log("-" * 100)
        positive = positive.sort_values('ROI%', ascending=False)
        for _, row in positive.head(20).iterrows():
            log(f"{row['Tier']:10s} | {row['Price']:8s} | Box:{str(row['Box']):3s} | "
                f"PrevWin:{str(row['PrevWon']):5s} | Fav:{row['FavRange']:4s} | "
                f"{row['Bets']:4d} ({row['BetsPerDay']:.1f}/d) | {row['Strike%']:.1f}% | ROI {row['ROI%']:+.1f}%")
    else:
        log("\nNo positive ROI configurations found at 100+ bets sample size")
    
    # Show best configurations regardless of ROI
    log("\nBest configurations by ROI (any sample size >= 100):")
    log("-" * 100)
    results_df = results_df[results_df['Bets'] >= 100].sort_values('ROI%', ascending=False)
    for _, row in results_df.head(15).iterrows():
        log(f"{row['Tier']:10s} | {row['Price']:8s} | Box:{str(row['Box']):3s} | "
            f"PrevWin:{str(row['PrevWon']):5s} | Fav:{row['FavRange']:4s} | "
            f"{row['Bets']:4d} ({row['BetsPerDay']:.1f}/d) | {row['Strike%']:.1f}% | ROI {row['ROI%']:+.1f}%")

def main():
    log("="*80)
    log("SITUATION ANALYSIS - Finding specific profitable patterns")
    log("="*80)
    
    df = load_data()
    log(f"Loaded {len(df):,} entries")
    
    # Convert types
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    
    # Remove nulls
    df = df.dropna(subset=['Position', 'StartingPrice'])
    
    analyze_box_performance(df)
    analyze_race_number(df)
    analyze_distance(df)
    analyze_market_conditions(df)
    analyze_form_patterns(df)
    analyze_track_tier_value(df)
    analyze_multivariate_combinations(df)
    
    log("\n" + "="*80)
    log("ANALYSIS COMPLETE")
    log("="*80)

if __name__ == "__main__":
    main()
