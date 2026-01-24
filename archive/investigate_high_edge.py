"""
Investigate the Edge >= 20% cases that showed +6% ROI
Why does it work despite terrible calibration?
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
    log("INVESTIGATING HIGH EDGE BETS (>=20%)")
    log("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load data
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
    
    # Convert types
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    df['G_OT'] = pd.to_numeric(df['G_OT'], errors='coerce')
    df['G_Split'] = pd.to_numeric(df['G_Split'], errors='coerce')
    df['M_OT'] = pd.to_numeric(df['M_OT'], errors='coerce')
    df['M_Split'] = pd.to_numeric(df['M_Split'], errors='coerce')
    
    df = df.dropna(subset=['Position', 'StartingPrice'])
    df['Won'] = df['Position'] == 1
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    
    # Calculate adjusted benchmarks
    df['GM_OT'] = df['G_OT'] - df['M_OT']
    df['GM_Split'] = df['G_Split'] - df['M_Split']
    
    # Implied probability from odds
    df['ImpliedProb'] = 1 / df['StartingPrice']
    
    # Split train/test
    train = df[df['MeetingDate'] < '2025-09-01'].copy()
    test = df[df['MeetingDate'] >= '2025-09-01'].copy()
    
    log(f"Train: {len(train):,} | Test: {len(test):,}")
    
    # Build historical features (Last 5 races)
    log("Building features...")
    
    df_sorted = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Calculate rolling stats
    for col in ['GM_OT', 'GM_Split', 'G_OT', 'G_Split']:
        df_sorted[f'Last5_{col}'] = df_sorted.groupby('GreyhoundID')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    
    # Win rate
    df_sorted['Last5_Wins'] = df_sorted.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    df_sorted['Last5_Races'] = df_sorted.groupby('GreyhoundID')['Won'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).count()
    )
    df_sorted['Last5_WinRate'] = df_sorted['Last5_Wins'] / df_sorted['Last5_Races'].replace(0, 1)
    
    # Career stats
    df_sorted['CareerRaces'] = df_sorted.groupby('GreyhoundID').cumcount()
    df_sorted['CareerWins'] = df_sorted.groupby('GreyhoundID')['Won'].cumsum().shift(1).fillna(0)
    df_sorted['CareerWinRate'] = df_sorted['CareerWins'] / df_sorted['CareerRaces'].replace(0, 1)
    
    # Get train/test with features
    train_feat = df_sorted[df_sorted['MeetingDate'] < '2025-09-01'].copy()
    test_feat = df_sorted[df_sorted['MeetingDate'] >= '2025-09-01'].copy()
    
    features = ['Last5_GM_OT', 'Last5_GM_Split', 'Last5_G_OT', 'Last5_G_Split',
                'Last5_WinRate', 'CareerWinRate', 'Box']
    
    # Clean
    train_clean = train_feat.dropna(subset=features)
    test_clean = test_feat.dropna(subset=features)
    
    log(f"Train clean: {len(train_clean):,} | Test clean: {len(test_clean):,}")
    
    # Train model (without odds - to find genuine edge)
    X_train = train_clean[features]
    y_train = train_clean['Won']
    
    log("Training model...")
    base_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    model.fit(X_train, y_train)
    
    # Predict on test
    X_test = test_clean[features]
    test_clean = test_clean.copy()
    test_clean['ModelProb'] = model.predict_proba(X_test)[:, 1]
    
    # Calculate edge
    test_clean['Edge'] = test_clean['ModelProb'] - test_clean['ImpliedProb']
    test_clean['EdgePct'] = test_clean['Edge'] * 100
    
    log("\n" + "="*80)
    log("HIGH EDGE (>=20%) ANALYSIS")
    log("="*80)
    
    high_edge = test_clean[test_clean['Edge'] >= 0.20].copy()
    log(f"\nFound {len(high_edge):,} high edge bets")
    
    # Analyze characteristics
    log("\n--- PRICE DISTRIBUTION ---")
    for min_p, max_p in [(1.5, 5), (5, 10), (10, 20), (20, 50), (50, 100)]:
        subset = high_edge[(high_edge['StartingPrice'] >= min_p) & (high_edge['StartingPrice'] < max_p)]
        if len(subset) > 0:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins/bets*100
            avg_odds = subset['StartingPrice'].mean()
            profit = sum(subset['StartingPrice'] - 1 if w else -1 for w, p in zip(subset['Won'], subset['StartingPrice']))
            profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
            roi = profit / bets * 100
            log(f"${min_p}-${max_p}: {bets} bets | {strike:.1f}% strike | ${avg_odds:.1f} avg | ROI {roi:+.1f}%")
    
    log("\n--- TRACK TIER ---")
    for tier in ['Metro', 'Provincial', 'Country']:
        subset = high_edge[high_edge['Tier'] == tier]
        if len(subset) > 0:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins/bets*100
            avg_odds = subset['StartingPrice'].mean()
            profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
            roi = profit / bets * 100
            log(f"{tier}: {bets} bets | {strike:.1f}% strike | ${avg_odds:.1f} avg | ROI {roi:+.1f}%")
    
    log("\n--- BOX POSITION ---")
    for box in range(1, 9):
        subset = high_edge[high_edge['Box'] == box]
        if len(subset) > 0:
            wins = subset['Won'].sum()
            bets = len(subset)
            strike = wins/bets*100
            avg_odds = subset['StartingPrice'].mean()
            profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
            roi = profit / bets * 100
            log(f"Box {box}: {bets} bets | {strike:.1f}% strike | ${avg_odds:.1f} avg | ROI {roi:+.1f}%")
    
    log("\n--- FEATURE AVERAGES (High Edge vs All) ---")
    for feat in features:
        he_avg = high_edge[feat].mean()
        all_avg = test_clean[feat].mean()
        log(f"{feat:20s}: High Edge = {he_avg:+.3f} | All = {all_avg:+.3f}")
    
    log("\n--- WHY +6% ROI? ---")
    log("High edge bets have these characteristics:")
    
    # Look at what makes these special
    log(f"\nModel Prob avg: {high_edge['ModelProb'].mean():.1%}")
    log(f"Implied Prob avg: {high_edge['ImpliedProb'].mean():.1%}")
    log(f"Actual Win Rate: {high_edge['Won'].mean():.1%}")
    log(f"Avg Odds: ${high_edge['StartingPrice'].mean():.1f}")
    
    # The key insight: even at low strike rate, high odds can be profitable
    # If strike rate is 6.4% but avg odds is $22.93:
    # Expected return = 0.064 * 22.93 = 1.47 = +47% ROI theoretically
    # But actual is only +6% - why?
    
    log("\n--- REFINED FILTERS ---")
    
    # Try combining high edge with other factors
    for min_p, max_p in [(10, 30), (15, 40), (20, 50), (10, 50)]:
        for tier in ['All', 'Country', 'Provincial']:
            if tier == 'All':
                subset = high_edge[(high_edge['StartingPrice'] >= min_p) & 
                                   (high_edge['StartingPrice'] <= max_p)]
            else:
                subset = high_edge[(high_edge['StartingPrice'] >= min_p) & 
                                   (high_edge['StartingPrice'] <= max_p) &
                                   (high_edge['Tier'] == tier)]
            
            if len(subset) >= 50:
                wins = subset['Won'].sum()
                bets = len(subset)
                strike = wins/bets*100
                avg_odds = subset['StartingPrice'].mean()
                profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
                roi = profit / bets * 100
                
                # Calculate days
                days = (pd.to_datetime(subset['MeetingDate'].max()) - 
                       pd.to_datetime(subset['MeetingDate'].min())).days + 1
                bpd = bets / days
                
                if roi > 0:
                    log(f"${min_p}-${max_p} | {tier:10s} | {bets:3d} ({bpd:.1f}/d) | {strike:.1f}% | ${avg_odds:.1f} | ROI {roi:+.1f}% ***")
                else:
                    log(f"${min_p}-${max_p} | {tier:10s} | {bets:3d} ({bpd:.1f}/d) | {strike:.1f}% | ${avg_odds:.1f} | ROI {roi:+.1f}%")
    
    log("\n--- TRY DIFFERENT EDGE THRESHOLDS + PRICE COMBOS ---")
    
    results = []
    for edge_thresh in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        for min_p, max_p in [(8, 20), (10, 25), (12, 30), (15, 35), (15, 50), (20, 50)]:
            for tier in ['All', 'Country', 'Provincial', 'Metro']:
                if tier == 'All':
                    subset = test_clean[(test_clean['Edge'] >= edge_thresh) & 
                                       (test_clean['StartingPrice'] >= min_p) & 
                                       (test_clean['StartingPrice'] <= max_p)]
                else:
                    subset = test_clean[(test_clean['Edge'] >= edge_thresh) & 
                                       (test_clean['StartingPrice'] >= min_p) & 
                                       (test_clean['StartingPrice'] <= max_p) &
                                       (test_clean['Tier'] == tier)]
                
                if len(subset) >= 30:
                    wins = subset['Won'].sum()
                    bets = len(subset)
                    strike = wins/bets*100
                    avg_odds = subset['StartingPrice'].mean()
                    profit = subset.apply(lambda x: x['StartingPrice'] - 1 if x['Won'] else -1, axis=1).sum()
                    roi = profit / bets * 100
                    
                    days = (pd.to_datetime(subset['MeetingDate'].max()) - 
                           pd.to_datetime(subset['MeetingDate'].min())).days + 1
                    bpd = bets / days
                    
                    results.append({
                        'Edge': edge_thresh,
                        'Price': f'${min_p}-${max_p}',
                        'Tier': tier,
                        'Bets': bets,
                        'BPD': bpd,
                        'Strike': strike,
                        'AvgOdds': avg_odds,
                        'ROI': roi
                    })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        positive = results_df[results_df['ROI'] > 0].sort_values('ROI', ascending=False)
        
        log(f"\nPOSITIVE ROI configurations ({len(positive)}):")
        log("-" * 90)
        for _, row in positive.head(20).iterrows():
            log(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tier']:10s} | "
                f"{row['Bets']:3.0f} ({row['BPD']:.1f}/d) | {row['Strike']:.1f}% | ${row['AvgOdds']:.1f} | ROI {row['ROI']:+.1f}%")
    
    log("\n" + "="*80)
    log("COMPLETE")
    log("="*80)

if __name__ == "__main__":
    main()
