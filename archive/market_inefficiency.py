"""
MARKET INEFFICIENCY ANALYSIS
Find when/where the market is less efficient
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

DB_PATH = 'greyhound_racing.db'

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_track_tier(track):
    metro = ['Wentworth Park', 'The Meadows', 'Albion Park', 'Sandown Park', 'Cannington']
    provincial = ['Richmond', 'Bulli', 'Shepparton', 'Ipswich', 'Dapto', 'Gosford', 'Warragul']
    if track in metro:
        return 'Metro'
    elif track in provincial:
        return 'Provincial'
    else:
        return 'Country'

def main():
    log("="*80)
    log("MARKET INEFFICIENCY ANALYSIS")
    log("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
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
    
    log(f"Loaded {len(df):,} entries")
    
    for col in ['Position', 'StartingPrice', 'Box', 'G_OT', 'G_Split', 'M_OT', 'M_Split', 'RaceNumber']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Position', 'StartingPrice'])
    df['Won'] = df['Position'] == 1
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    df['ImpliedProb'] = 1 / df['StartingPrice']
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek  # 0=Mon, 6=Sun
    df['DayName'] = df['MeetingDate'].dt.day_name()
    
    # Focus on 2025 test data
    test = df[df['MeetingDate'] >= '2025-01-01'].copy()
    log(f"2025 data: {len(test):,} entries")
    
    log("\n" + "="*80)
    log("1. DAY OF WEEK ANALYSIS (Value Dogs $4-$15)")
    log("="*80)
    
    value_test = test[(test['StartingPrice'] >= 4) & (test['StartingPrice'] <= 15)].copy()
    value_test['Profit'] = value_test.apply(
        lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1
    )
    
    day_stats = value_test.groupby('DayName').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    })
    day_stats.columns = ['Wins', 'Bets', 'Strike', 'Profit', 'AvgOdds']
    day_stats['ROI'] = day_stats['Profit'] / day_stats['Bets'] * 100
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = day_stats.reindex(day_order)
    
    log("\nDay of Week (Value dogs $4-$15):")
    log("-" * 60)
    for day in day_order:
        if day in day_stats.index:
            row = day_stats.loc[day]
            log(f"{day:10s}: {row['Bets']:.0f} bets | {row['Strike']*100:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    
    log("\n" + "="*80)
    log("2. TIME OF DAY (Race Number as proxy)")
    log("="*80)
    
    race_stats = value_test.groupby('RaceNumber').agg({
        'Won': ['sum', 'count', 'mean'],
        'Profit': 'sum',
        'StartingPrice': 'mean'
    })
    race_stats.columns = ['Wins', 'Bets', 'Strike', 'Profit', 'AvgOdds']
    race_stats['ROI'] = race_stats['Profit'] / race_stats['Bets'] * 100
    
    log("\nRace Number (Value dogs $4-$15):")
    log("-" * 60)
    for rn in sorted(race_stats.index):
        if rn <= 12 and race_stats.loc[rn, 'Bets'] >= 500:
            row = race_stats.loc[rn]
            marker = "***" if row['ROI'] > -15 else ""
            log(f"Race {rn:2.0f}: {row['Bets']:.0f} bets | {row['Strike']*100:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}% {marker}")
    
    log("\n" + "="*80)
    log("3. COMPETITIVE RACES (Close prices)")
    log("="*80)
    
    # Calculate race competitiveness (gap between fav and 2nd fav)
    fav_by_race = test.groupby('RaceID')['StartingPrice'].agg(['min', lambda x: x.nsmallest(2).iloc[-1] if len(x) >= 2 else x.min()])
    fav_by_race.columns = ['FavPrice', 'SecondPrice']
    fav_by_race['PriceGap'] = fav_by_race['SecondPrice'] - fav_by_race['FavPrice']
    fav_by_race['Competitive'] = fav_by_race['PriceGap'] < 0.5  # Close race
    
    test = test.merge(fav_by_race, on='RaceID')
    
    # Value dogs in competitive vs non-competitive
    for comp in [True, False]:
        subset = test[(test['StartingPrice'] >= 4) & 
                      (test['StartingPrice'] <= 15) & 
                      (test['Competitive'] == comp)]
        if len(subset) > 0:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins / bets * 100
            profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
            roi = profit / bets * 100
            avg_odds = subset['StartingPrice'].mean()
            log(f"{'Competitive' if comp else 'Non-competitive':15s}: {bets:,} bets | {strike:.1f}% | ${avg_odds:.1f} | ROI {roi:+.1f}%")
    
    log("\n" + "="*80)
    log("4. OVERROUND ANALYSIS")
    log("="*80)
    
    # Calculate market overround per race
    overround = test.groupby('RaceID')['ImpliedProb'].sum().reset_index()
    overround.columns = ['RaceID', 'Overround']
    test = test.merge(overround, on='RaceID')
    
    # Categorize overround
    test['OverroundCat'] = pd.cut(test['Overround'], 
                                   bins=[0, 1.1, 1.2, 1.3, 1.4, 2.0],
                                   labels=['<110%', '110-120%', '120-130%', '130-140%', '>140%'])
    
    log("\nValue dogs $4-$15 by Market Overround:")
    log("-" * 60)
    
    value_test = test[(test['StartingPrice'] >= 4) & (test['StartingPrice'] <= 15)].copy()
    
    for cat in ['<110%', '110-120%', '120-130%', '130-140%', '>140%']:
        subset = value_test[value_test['OverroundCat'] == cat]
        if len(subset) >= 100:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins / bets * 100
            profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
            roi = profit / bets * 100
            avg_odds = subset['StartingPrice'].mean()
            marker = "***" if roi > -15 else ""
            log(f"{cat:12s}: {bets:,} bets | {strike:.1f}% | ${avg_odds:.1f} | ROI {roi:+.1f}% {marker}")
    
    log("\n" + "="*80)
    log("5. MULTI-FACTOR INEFFICIENCY SEARCH")
    log("="*80)
    
    results = []
    
    for day in [None, 0, 1, 2, 3, 4, 5, 6]:  # None = all days
        for early_race in [None, True, False]:  # Early = race 1-4
            for tier in ['All', 'Country', 'Provincial']:
                for overround_max in [1.2, 1.3, 1.4, None]:
                    for min_p, max_p in [(3, 8), (4, 12), (5, 15), (8, 20)]:
                        
                        subset = test[(test['StartingPrice'] >= min_p) & 
                                     (test['StartingPrice'] <= max_p)]
                        
                        if day is not None:
                            subset = subset[subset['DayOfWeek'] == day]
                        
                        if early_race is not None:
                            if early_race:
                                subset = subset[subset['RaceNumber'] <= 4]
                            else:
                                subset = subset[subset['RaceNumber'] >= 9]
                        
                        if tier != 'All':
                            subset = subset[subset['Tier'] == tier]
                        
                        if overround_max is not None:
                            subset = subset[subset['Overround'] <= overround_max]
                        
                        if len(subset) < 100:
                            continue
                        
                        wins = subset['Won'].sum()
                        bets = len(subset)
                        strike = wins / bets * 100
                        profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
                        roi = profit / bets * 100
                        avg_odds = subset['StartingPrice'].mean()
                        
                        days_total = (test['MeetingDate'].max() - test['MeetingDate'].min()).days + 1
                        bpd = bets / days_total
                        
                        results.append({
                            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day] if day is not None else 'All',
                            'RaceTime': 'Early' if early_race == True else ('Late' if early_race == False else 'All'),
                            'Tier': tier,
                            'Overround': f'<={overround_max*100:.0f}%' if overround_max else 'All',
                            'Price': f'${min_p}-${max_p}',
                            'Bets': bets,
                            'BPD': bpd,
                            'Strike': strike,
                            'AvgOdds': avg_odds,
                            'ROI': roi
                        })
    
    results_df = pd.DataFrame(results)
    
    # Show best configs
    positive = results_df[results_df['ROI'] > 0].sort_values('ROI', ascending=False)
    
    if len(positive) > 0:
        log(f"\nPOSITIVE ROI configurations ({len(positive)}):")
        log("-" * 110)
        for _, row in positive.head(20).iterrows():
            log(f"{row['Day']:3s} | {row['RaceTime']:5s} | {row['Tier']:10s} | OR:{row['Overround']:7s} | "
                f"{row['Price']:8s} | {row['Bets']:4.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    else:
        log("\nNo positive ROI configurations found")
    
    # Best at volume
    log(f"\nBest configurations with 3+ bets/day:")
    log("-" * 110)
    high_vol = results_df[(results_df['BPD'] >= 3)].sort_values('ROI', ascending=False).head(10)
    for _, row in high_vol.iterrows():
        log(f"{row['Day']:3s} | {row['RaceTime']:5s} | {row['Tier']:10s} | OR:{row['Overround']:7s} | "
            f"{row['Price']:8s} | {row['Bets']:4.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    
    log("\n" + "="*80)
    log("COMPLETE")
    log("="*80)

if __name__ == "__main__":
    main()
