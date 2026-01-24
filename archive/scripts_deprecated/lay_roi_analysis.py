"""
Lay ROI Calculator - Proper Accounting
=======================================
For laying:
- If dog LOSES: You WIN the stake (1 unit profit)
- If dog WINS: You LOSE (BSP - 1) units (the liability)

Lay ROI = (Lay Wins × 1 - Lay Losses × (Avg BSP - 1)) / Total Bets
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

def run_lay_analysis():
    print("="*70)
    print("LAY ROI ANALYSIS - PROPER LIABILITY ACCOUNTING")
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
        
        if race_date < datetime(2024, 1, 1):
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
            continue
        
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
                    'Won': r['Won'],
                    'BSP': r['BSP'],
                    'Tier': tier
                })
        
        if len(race_features) >= 4:
            X = np.array(race_features)
            X_scaled = scaler.transform(X)
            raw_probs = model.predict_proba(X_scaled)[:, 1]
            
            total_prob = raw_probs.sum()
            norm_probs = raw_probs / total_prob if total_prob > 0 else raw_probs
            
            for i, dog in enumerate(race_dogs):
                if pd.notna(dog['BSP']) and dog['BSP'] > 1:
                    rated_prob = norm_probs[i]
                    value = (dog['BSP'] * rated_prob) - 1
                    
                    all_bets.append({
                        'Won': dog['Won'],
                        'BSP': dog['BSP'],
                        'Tier': dog['Tier'],
                        'Value': value
                    })
        
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
    print(f"  Total rated: {len(bets_df):,}")
    
    print("\n[4/4] LAY ROI Results...")
    print("="*70)
    
    def calc_lay_roi(df, value_thresh, odds_min, odds_max, label):
        """
        Lay bet P&L:
        - If dog LOSES (we win lay): +1 unit profit
        - If dog WINS (we lose lay): -(BSP - 1) units loss
        """
        subset = df[(df['Value'] <= value_thresh) & (df['BSP'] >= odds_min) & (df['BSP'] <= odds_max)]
        if len(subset) < 50:
            print(f"{label}: N/A (n={len(subset)})")
            return
        
        n_bets = len(subset)
        n_wins = subset['Won'].sum()  # Dog wins = we lose
        n_losses = n_bets - n_wins     # Dog loses = we win
        
        # Calculate P&L
        lay_wins_profit = n_losses * 1.0  # We win 1 unit when dog loses
        lay_losses = subset[subset['Won'] == 1]['BSP'] - 1  # We lose (BSP-1) when dog wins
        lay_losses_total = lay_losses.sum()
        
        total_profit = lay_wins_profit - lay_losses_total
        roi = total_profit / n_bets * 100
        
        avg_bsp = subset['BSP'].mean()
        lay_sr = n_losses / n_bets * 100
        
        print(f"{label}: {n_bets:,} lays, {n_losses:,} wins ({lay_sr:.1f}%), Avg BSP: ${avg_bsp:.2f}, Profit: {total_profit:.1f}u, ROI: {roi:+.1f}%")
    
    print("\n--- LAY ROI BY VALUE THRESHOLD ($2-$10) ---")
    for thresh in [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]:
        calc_lay_roi(bets_df, thresh, 2, 10, f"Value <= {thresh*100:.0f}%")
    
    print("\n--- LAY ROI BY ODDS RANGE (Value <= -15%) ---")
    for low, high in [(1.5, 3), (2, 4), (3, 6), (4, 8), (6, 10)]:
        calc_lay_roi(bets_df, -0.15, low, high, f"${low}-${high}")
    
    print("\n--- LAY ROI BY TIER (Value <= -15%, $2-$10) ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        subset = bets_df[(bets_df['Tier'] == tier)]
        calc_lay_roi(subset, -0.15, 2, 10, name)
    
    print("\n--- COMPARISON: BACK vs LAY ---")
    print("\nBACK (Value >= 10%, $5-$10):")
    back = bets_df[(bets_df['Value'] >= 0.10) & (bets_df['BSP'] >= 5) & (bets_df['BSP'] <= 10)]
    if len(back) >= 50:
        wins = back['Won'].sum()
        returns = back[back['Won'] == 1]['BSP'].sum()
        profit = returns - len(back)
        roi = profit / len(back) * 100
        print(f"  {len(back):,} bets, {wins:,} wins ({wins/len(back)*100:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
    
    print("\nLAY (Value <= -15%, $2-$5):")
    calc_lay_roi(bets_df, -0.15, 2, 5, "  ")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_lay_analysis()
