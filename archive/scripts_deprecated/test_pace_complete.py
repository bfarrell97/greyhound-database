"""
Pace Model - Only Bet on "Complete Data" Races
===============================================
Only bet when ALL dogs in the race have 3+ races with valid times.
This ensures fair leader selection and filters to predictable dogs.
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

def run_test():
    print("="*70)
    print("PACE MODEL - COMPLETE DATA RACES ONLY")
    print("="*70)
    print("Only betting on races where ALL dogs have 3+ historical races")
    
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
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[3/4] Building predictions...")
    
    dog_history = {}
    predictions = []
    complete_races = 0
    incomplete_races = 0
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
        
        # Check if ALL dogs have 3+ races with valid times
        all_have_data = True
        race_features = []
        
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            hist = dog_history.get(dog_id, [])
            recent_times = [h[1] for h in hist[-5:] if h[1] is not None and not np.isnan(h[1])]
            
            if len(recent_times) < 3:
                all_have_data = False
                break
            
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
            
            race_features.append({
                'GreyhoundID': dog_id,
                'Won': r['Won'],
                'BSP': r['BSP'],
                'Tier': tier,
                'Features': feats
            })
        
        if all_have_data and len(race_features) >= 4:
            complete_races += 1
            
            # Predict probabilities for all dogs
            X = np.array([rf['Features'] for rf in race_features])
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[:, 1]
            
            # Find leader (highest probability)
            leader_idx = np.argmax(probs)
            leader = race_features[leader_idx]
            
            predictions.append({
                'RaceID': race_id,
                'Won': leader['Won'],
                'BSP': leader['BSP'],
                'Tier': leader['Tier'],
                'Prob': probs[leader_idx]
            })
        else:
            incomplete_races += 1
        
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
    print(f"  Complete data races (testable): {complete_races:,}")
    print(f"  Incomplete data races (skipped): {incomplete_races:,}")
    
    pred_df = pd.DataFrame(predictions)
    pred_df['BSP'] = pd.to_numeric(pred_df['BSP'], errors='coerce')
    pred_df = pred_df.dropna(subset=['BSP'])
    
    print(f"\n[4/4] Results...")
    print("="*70)
    
    def backtest(df, label):
        if len(df) < 50:
            print(f"{label}: N/A (n={len(df)})")
            return
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        returns = df[df['Won'] == 1]['BSP'].sum()
        profit = returns - len(df)
        roi = profit / len(df) * 100
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- ALL TRACKS ---")
    backtest(pred_df, "All")
    backtest(pred_df[(pred_df['BSP'] >= 2) & (pred_df['BSP'] <= 10)], "$2-$10")
    backtest(pred_df[(pred_df['BSP'] >= 3) & (pred_df['BSP'] <= 8)], "$3-$8")
    
    print("\n--- BY TIER ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        t = pred_df[(pred_df['Tier'] == tier) & (pred_df['BSP'] >= 2) & (pred_df['BSP'] <= 10)]
        backtest(t, f"{name} $2-$10")
    
    print("\n--- BY CONFIDENCE ---")
    high_conf = pred_df[pred_df['Prob'] >= pred_df['Prob'].quantile(0.75)]
    backtest(high_conf[(high_conf['BSP'] >= 2) & (high_conf['BSP'] <= 10)], "Top 25% confidence $2-$10")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_test()
