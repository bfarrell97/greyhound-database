"""
Pace Model - Rated Prices & Value Betting
==========================================
Uses Pace ML model to generate rated odds for each dog,
compares to BSP, and finds value bets.

Value = (BSP × Model Probability) - 1
Bet when Value > threshold
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def run_rated_prices():
    print("="*70)
    print("PACE MODEL - RATED PRICES & VALUE BETTING")
    print("="*70)
    
    # Load pace model
    print("\n[1/4] Loading Pace model...")
    with open('models/pace_gb_model_lite.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model = artifacts['model']
    scaler = artifacts['scaler']
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    print("\n[2/4] Loading data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2019-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    df['FinishTime'] = pd.to_numeric(df['FinishTime'], errors='coerce')
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[3/4] Building rated prices...")
    
    dog_history = {}
    all_bets = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        # Only process test period
        if race_date < datetime(2024, 1, 1):
            # Update history only
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                if dog_id not in dog_history:
                    dog_history[dog_id] = []
                dog_history[dog_id].append((
                    race_date,
                    r['NormTime'] if pd.notna(r['NormTime']) else None,
                    r['Split'] if pd.notna(r['Split']) else None,
                    r['Position']
                ))
            processed += 1
            if processed % 50000 == 0:
                print(f"  {processed:,} races (building history)...")
            continue
        
        # Build features for dogs with history
        race_features = []
        race_dogs = []
        
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            hist = dog_history.get(dog_id, [])
            recent_times = [h[1] for h in hist[-5:] if h[1] is not None and not np.isnan(h[1])]
            
            if len(recent_times) >= 3:
                recent_splits = [h[2] for h in hist[-5:] if h[2] is not None and not np.isnan(h[2])]
                recent_pos = [h[3] for h in hist[-5:] if h[3] is not None]
                
                time_lag1 = recent_times[-1]
                time_lag2 = recent_times[-2] if len(recent_times) >= 2 else time_lag1
                time_lag3 = recent_times[-3] if len(recent_times) >= 3 else time_lag2
                time_avg3 = np.mean(recent_times[-3:])
                time_avg5 = np.mean(recent_times[-5:]) if len(recent_times) >= 5 else time_avg3
                time_std = np.std(recent_times[-5:]) if len(recent_times) >= 3 else 0
                time_trend = time_lag1 - time_lag3
                time_best = min(recent_times)
                split_avg = np.mean(recent_splits) if recent_splits else 0
                split_lag1 = recent_splits[-1] if recent_splits else 0
                pos_avg = np.mean(recent_pos) if recent_pos else 4
                pos_lag1 = recent_pos[-1] if recent_pos else 4
                win_rate = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                place_rate = sum(1 for p in recent_pos if p <= 3) / len(recent_pos) if recent_pos else 0
                last_date = hist[-1][0]
                days_since = min((race_date - last_date).days, 90)
                race_count = min(len(hist), 50)
                
                feats = [time_lag1, time_lag2, time_lag3, time_avg3, time_avg5,
                         time_std, time_trend, time_best, split_avg, split_lag1,
                         pos_avg, pos_lag1, win_rate, place_rate,
                         days_since, race_count, r['Box'], distance, tier]
                
                race_features.append(feats)
                race_dogs.append({
                    'GreyhoundID': dog_id,
                    'Won': r['Won'],
                    'BSP': r['BSP'],
                    'Tier': tier
                })
        
        # Need at least 4 dogs with features for fair rating
        if len(race_features) >= 4:
            # Predict raw probabilities
            X = np.array(race_features)
            X_scaled = scaler.transform(X)
            raw_probs = model.predict_proba(X_scaled)[:, 1]
            
            # Normalize to sum to 1 (proper race probabilities)
            total_prob = raw_probs.sum()
            if total_prob > 0:
                norm_probs = raw_probs / total_prob
            else:
                norm_probs = raw_probs
            
            # Calculate rated odds and value for each dog
            for i, dog in enumerate(race_dogs):
                if pd.notna(dog['BSP']) and dog['BSP'] > 1:
                    rated_prob = norm_probs[i]
                    rated_odds = 1 / rated_prob if rated_prob > 0 else 100
                    
                    # Value = expected return - 1
                    value = (dog['BSP'] * rated_prob) - 1
                    
                    all_bets.append({
                        'RaceID': race_id,
                        'Won': dog['Won'],
                        'BSP': dog['BSP'],
                        'Tier': dog['Tier'],
                        'RatedProb': rated_prob,
                        'RatedOdds': rated_odds,
                        'Value': value
                    })
        
        # Update history
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            if dog_id not in dog_history:
                dog_history[dog_id] = []
            dog_history[dog_id].append((
                race_date,
                r['NormTime'] if pd.notna(r['NormTime']) else None,
                r['Split'] if pd.notna(r['Split']) else None,
                r['Position']
            ))
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    
    bets_df = pd.DataFrame(all_bets)
    print(f"  Total rated bets: {len(bets_df):,}")
    
    print("\n[4/4] Value Betting Results...")
    print("="*70)
    
    def test_value(df, threshold, odds_min, odds_max, label):
        subset = df[(df['Value'] >= threshold) & (df['BSP'] >= odds_min) & (df['BSP'] <= odds_max)]
        if len(subset) < 50:
            print(f"{label}: N/A (n={len(subset)})")
            return
        wins = subset['Won'].sum()
        sr = wins / len(subset) * 100
        returns = subset[subset['Won'] == 1]['BSP'].sum()
        profit = returns - len(subset)
        roi = profit / len(subset) * 100
        print(f"{label}: {len(subset):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- VALUE THRESHOLDS ($2-$20) ---")
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        test_value(bets_df, thresh, 2, 20, f"Value >= {thresh*100:.0f}%")
    
    print("\n--- VALUE >= 10% BY ODDS RANGE ---")
    for low, high in [(2, 5), (3, 8), (5, 10), (10, 20)]:
        test_value(bets_df, 0.10, low, high, f"${low}-${high}")
    
    print("\n--- VALUE >= 15% BY TIER ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        subset = bets_df[(bets_df['Tier'] == tier) & (bets_df['Value'] >= 0.15) & 
                         (bets_df['BSP'] >= 2) & (bets_df['BSP'] <= 20)]
        if len(subset) >= 50:
            wins = subset['Won'].sum()
            sr = wins / len(subset) * 100
            returns = subset[subset['Won'] == 1]['BSP'].sum()
            profit = returns - len(subset)
            roi = profit / len(subset) * 100
            print(f"{name}: {len(subset):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- NEGATIVE VALUE (LAY OPPORTUNITIES) ---")
    for thresh in [-0.10, -0.20, -0.30]:
        subset = bets_df[(bets_df['Value'] <= thresh) & (bets_df['BSP'] >= 2) & (bets_df['BSP'] <= 10)]
        if len(subset) >= 50:
            wins = subset['Won'].sum()
            sr = wins / len(subset) * 100
            print(f"Value <= {thresh*100:.0f}%: {len(subset):,} dogs, {wins:,} wins ({sr:.1f}%) - LAY WIN RATE: {100-sr:.1f}%")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_rated_prices()
