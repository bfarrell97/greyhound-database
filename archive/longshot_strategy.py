"""
LONGSHOT VALUE BETTING SYSTEM
Based on finding: High edge + Country + $15-$50 = profitable

Target: 3-10 bets/day with positive ROI
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
    log("LONGSHOT VALUE BETTING - Full Backtest")
    log("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load ALL data for longer backtest
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
    WHERE rm.MeetingDate >= '2023-01-01'
        AND ge.Position IS NOT NULL
        AND ge.StartingPrice IS NOT NULL
        AND ge.StartingPrice > 0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    log(f"Loaded {len(df):,} entries")
    
    # Convert types
    for col in ['Position', 'StartingPrice', 'Box', 'G_OT', 'G_Split', 'M_OT', 'M_Split']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Position', 'StartingPrice'])
    df['Won'] = df['Position'] == 1
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    df['GM_OT'] = df['G_OT'] - df['M_OT']
    df['GM_Split'] = df['G_Split'] - df['M_Split']
    df['ImpliedProb'] = 1 / df['StartingPrice']
    
    # Build features
    log("Building features...")
    df_sorted = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    for col in ['GM_OT', 'GM_Split', 'G_OT', 'G_Split']:
        df_sorted[f'Last5_{col}'] = df_sorted.groupby('GreyhoundID')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    
    df_sorted['Last5_Wins'] = df_sorted.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    df_sorted['Last5_Races'] = df_sorted.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).count()
    )
    df_sorted['Last5_WinRate'] = df_sorted['Last5_Wins'] / df_sorted['Last5_Races'].replace(0, 1)
    
    df_sorted['CareerRaces'] = df_sorted.groupby('GreyhoundID').cumcount()
    df_sorted['CareerWins'] = df_sorted.groupby('GreyhoundID')['Won'].cumsum().shift(1).fillna(0)
    df_sorted['CareerWinRate'] = df_sorted['CareerWins'] / df_sorted['CareerRaces'].replace(0, 1)
    
    features = ['Last5_GM_OT', 'Last5_GM_Split', 'Last5_G_OT', 'Last5_G_Split',
                'Last5_WinRate', 'CareerWinRate', 'Box']
    
    # Use 2023-2024 for training, 2025 for testing
    train_data = df_sorted[(df_sorted['MeetingDate'] >= '2023-01-01') & 
                           (df_sorted['MeetingDate'] < '2025-01-01')].copy()
    test_data = df_sorted[df_sorted['MeetingDate'] >= '2025-01-01'].copy()
    
    train_clean = train_data.dropna(subset=features)
    test_clean = test_data.dropna(subset=features)
    
    log(f"Train: {len(train_clean):,} (2023-2024)")
    log(f"Test: {len(test_clean):,} (2025)")
    
    # Train model
    log("Training model...")
    X_train = train_clean[features]
    y_train = train_clean['Won']
    
    base_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    model.fit(X_train, y_train)
    
    # Predict
    X_test = test_clean[features]
    test_clean = test_clean.copy()
    test_clean['ModelProb'] = model.predict_proba(X_test)[:, 1]
    test_clean['Edge'] = test_clean['ModelProb'] - test_clean['ImpliedProb']
    
    # Calculate test period days
    test_days = (pd.to_datetime(test_clean['MeetingDate'].max()) - 
                 pd.to_datetime(test_clean['MeetingDate'].min())).days + 1
    log(f"Test period: {test_days} days (Jan 1 - Dec 9, 2025)")
    
    log("\n" + "="*80)
    log("STRATEGY RESULTS - Full 2025 Backtest")
    log("="*80)
    
    # Test various configurations targeting 3-10 bets/day
    results = []
    
    for edge_thresh in [0.12, 0.15, 0.18, 0.20, 0.22]:
        for min_p, max_p in [(10, 30), (12, 35), (15, 40), (15, 50), (10, 50)]:
            for tiers in [['Country'], ['Country', 'Provincial'], ['All']]:
                for boxes in [None, [1, 2], [1, 2, 8], [1]]:
                    
                    # Apply filters
                    subset = test_clean[(test_clean['Edge'] >= edge_thresh) & 
                                       (test_clean['StartingPrice'] >= min_p) & 
                                       (test_clean['StartingPrice'] <= max_p)]
                    
                    if tiers != ['All']:
                        subset = subset[subset['Tier'].isin(tiers)]
                    
                    if boxes is not None:
                        subset = subset[subset['Box'].isin(boxes)]
                    
                    if len(subset) < 50:
                        continue
                    
                    wins = subset['Won'].sum()
                    bets = len(subset)
                    bpd = bets / test_days
                    strike = wins / bets * 100
                    avg_odds = subset['StartingPrice'].mean()
                    profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
                    roi = profit / bets * 100
                    
                    results.append({
                        'Edge': edge_thresh,
                        'Price': f'${min_p}-${max_p}',
                        'Tiers': '+'.join(tiers) if tiers != ['All'] else 'All',
                        'Boxes': str(boxes) if boxes else 'All',
                        'Bets': bets,
                        'BPD': bpd,
                        'Strike': strike,
                        'AvgOdds': avg_odds,
                        'Profit': profit,
                        'ROI': roi
                    })
    
    results_df = pd.DataFrame(results)
    
    # Filter to 3-10 bets per day with positive ROI
    target = results_df[(results_df['BPD'] >= 3) & (results_df['BPD'] <= 10) & (results_df['ROI'] > 0)]
    
    log(f"\nConfigurations with 3-10 bets/day and positive ROI ({len(target)}):")
    log("-" * 100)
    
    if len(target) > 0:
        target = target.sort_values('ROI', ascending=False)
        for _, row in target.iterrows():
            log(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tiers']:20s} | Box:{row['Boxes']:12s} | "
                f"{row['Bets']:4.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    else:
        log("No configurations found at 3-10 bets/day with positive ROI")
    
    # Show best overall configs
    log(f"\nBest configurations by ROI (any volume, min 100 bets):")
    log("-" * 100)
    
    best = results_df[results_df['Bets'] >= 100].sort_values('ROI', ascending=False).head(15)
    for _, row in best.iterrows():
        marker = "***" if row['ROI'] > 30 else ""
        log(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tiers']:20s} | Box:{row['Boxes']:12s} | "
            f"{row['Bets']:4.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}% {marker}")
    
    # Best at higher volume
    log(f"\nBest configurations with 2+ bets/day:")
    log("-" * 100)
    
    high_vol = results_df[(results_df['BPD'] >= 2) & (results_df['ROI'] > 0)].sort_values('ROI', ascending=False).head(10)
    for _, row in high_vol.iterrows():
        log(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tiers']:20s} | Box:{row['Boxes']:12s} | "
            f"{row['Bets']:4.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    
    # Look at different approach - combine longshots with shorter prices
    log("\n" + "="*80)
    log("COMBINED STRATEGY: Longshots + Shorter Prices")
    log("="*80)
    
    # Strategy 1: High edge longshots
    longshot = test_clean[(test_clean['Edge'] >= 0.18) & 
                          (test_clean['StartingPrice'] >= 15) & 
                          (test_clean['StartingPrice'] <= 50) &
                          (test_clean['Tier'].isin(['Country', 'Provincial']))]
    
    # Strategy 2: Moderate edge, shorter prices, Box 1
    short = test_clean[(test_clean['Edge'] >= 0.10) & 
                       (test_clean['StartingPrice'] >= 3) & 
                       (test_clean['StartingPrice'] <= 8) &
                       (test_clean['Box'] == 1)]
    
    # Combine
    combined = pd.concat([longshot, short]).drop_duplicates(subset=['RaceID', 'GreyhoundID'])
    
    if len(combined) > 0:
        wins = combined['Won'].sum()
        bets = len(combined)
        bpd = bets / test_days
        strike = wins / bets * 100
        avg_odds = combined['StartingPrice'].mean()
        profit = combined.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
        roi = profit / bets * 100
        
        log(f"\nLongshot strategy only: {len(longshot)} bets ({len(longshot)/test_days:.1f}/d)")
        ls_wins = longshot['Won'].sum()
        ls_profit = longshot.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
        log(f"  Strike: {ls_wins/len(longshot)*100:.1f}% | ROI: {ls_profit/len(longshot)*100:+.1f}%")
        
        log(f"\nShort price (Box 1) strategy: {len(short)} bets ({len(short)/test_days:.1f}/d)")
        sh_wins = short['Won'].sum()
        sh_profit = short.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
        log(f"  Strike: {sh_wins/len(short)*100:.1f}% | ROI: {sh_profit/len(short)*100:+.1f}%")
        
        log(f"\nCOMBINED: {bets} bets ({bpd:.1f}/day)")
        log(f"  Strike: {strike:.1f}% | Avg Odds: ${avg_odds:.1f} | ROI: {roi:+.1f}%")
    
    log("\n" + "="*80)
    log("COMPLETE")
    log("="*80)

if __name__ == "__main__":
    main()
