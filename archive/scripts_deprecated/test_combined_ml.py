"""
Combined Elo + Pace ML Model Test
==================================
Loads both trained models and tests combinations:
1. Average of probabilities
2. Both agree on top pick
3. Ensemble (max of both)
"""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def get_k(track):
    if track in METRO_TRACKS: return 40
    elif track in PROVINCIAL_TRACKS: return 32
    return 24

def run_combined():
    print("="*70)
    print("COMBINED ELO + PACE ML MODEL")
    print("="*70)
    
    # Load models
    print("\n[1/5] Loading trained models...")
    with open('models/elo_gb_model_lite.pkl', 'rb') as f:
        elo_artifacts = pickle.load(f)
    with open('models/pace_gb_model_lite.pkl', 'rb') as f:
        pace_artifacts = pickle.load(f)
    
    elo_model = elo_artifacts['model']
    elo_scaler = elo_artifacts['scaler']
    pace_model = pace_artifacts['model']
    pace_scaler = pace_artifacts['scaler']
    
    print("  Elo model loaded")
    print("  Pace model loaded")
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load benchmarks
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    # Load 2024-2025 test data
    print("\n[2/5] Loading test data (2024-2025)...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
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
    
    # Build features for both models
    print("\n[3/5] Building features...")
    
    # For Elo
    elo_ratings = defaultdict(lambda: 1500)
    elo_race_count = defaultdict(int)
    elo_win_count = defaultdict(int)
    elo_last_date = {}
    
    # For Pace
    pace_history = {}
    
    predictions = []
    processed = 0
    
    df = df.sort_values(['MeetingDate', 'RaceID'])
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        k = get_k(track)
        
        # === ELO FEATURES ===
        race_elo = {r['GreyhoundID']: elo_ratings[r['GreyhoundID']] for _, r in race_df.iterrows()}
        elo_sorted = sorted(race_elo.items(), key=lambda x: x[1], reverse=True)
        elo_leader = elo_sorted[0][0]
        elo_2nd = elo_sorted[1][0] if len(elo_sorted) > 1 else elo_leader
        total_exp = sum(np.exp(r / 400) for r in race_elo.values())
        elo_expected = {d: np.exp(race_elo[d] / 400) / total_exp for d in race_elo}
        
        # Test period only
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                
                # Elo features
                days_since_elo = min((race_date - elo_last_date.get(dog_id, race_date)).days, 90) if dog_id in elo_last_date else 60
                elo_rank = [i+1 for i, (d, _) in enumerate(elo_sorted) if d == dog_id][0]
                gap_to_leader = race_elo[elo_leader] - race_elo[dog_id]
                gap_to_2nd = race_elo[elo_2nd] - race_elo[dog_id] if dog_id != elo_leader else 0
                win_rate_elo = elo_win_count[dog_id] / elo_race_count[dog_id] if elo_race_count[dog_id] > 0 else 0
                
                elo_feats = np.array([[race_elo[dog_id], elo_rank, elo_expected[dog_id], 
                                       gap_to_leader, gap_to_2nd, 1 if dog_id == elo_leader else 0,
                                       days_since_elo, min(elo_race_count[dog_id], 50), win_rate_elo,
                                       min(elo_win_count[dog_id], 20), r['Box'], distance, tier, len(race_df)]])
                
                # Pace features
                hist = pace_history.get(dog_id, [])
                if len(hist) >= 3:
                    recent_times = [h[1] for h in hist[-5:] if h[1] is not None and not np.isnan(h[1])]
                    recent_splits = [h[2] for h in hist[-5:] if h[2] is not None and not np.isnan(h[2])]
                    recent_pos = [h[3] for h in hist[-5:] if h[3] is not None]
                    
                    if len(recent_times) >= 3:
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
                        win_rate_pace = sum(1 for p in recent_pos if p == 1) / len(recent_pos) if recent_pos else 0
                        place_rate = sum(1 for p in recent_pos if p <= 3) / len(recent_pos) if recent_pos else 0
                        last_date = hist[-1][0]
                        days_since_pace = min((race_date - last_date).days, 90)
                        race_count_pace = min(len(hist), 50)
                        
                        pace_feats = np.array([[time_lag1, time_lag2, time_lag3, time_avg3, time_avg5,
                                               time_std, time_trend, time_best, split_avg, split_lag1,
                                               pos_avg, pos_lag1, win_rate_pace, place_rate,
                                               days_since_pace, race_count_pace, r['Box'], distance, tier]])
                        
                        # Predict with both models
                        elo_prob = elo_model.predict_proba(elo_scaler.transform(elo_feats))[0, 1]
                        pace_prob = pace_model.predict_proba(pace_scaler.transform(pace_feats))[0, 1]
                        
                        predictions.append({
                            'RaceID': race_id,
                            'GreyhoundID': dog_id,
                            'Won': r['Won'],
                            'BSP': r['BSP'],
                            'Tier': tier,
                            'EloProb': elo_prob,
                            'PaceProb': pace_prob,
                            'AvgProb': (elo_prob + pace_prob) / 2,
                            'MaxProb': max(elo_prob, pace_prob)
                        })
        
        # Update Elo ratings
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            actual = 1.0 if r['Won'] else 0.0
            elo_ratings[dog_id] += k * (actual - elo_expected[dog_id])
            elo_race_count[dog_id] += 1
            if actual == 1.0:
                elo_win_count[dog_id] += 1
            elo_last_date[dog_id] = race_date
            
            # Update pace history
            if dog_id not in pace_history:
                pace_history[dog_id] = []
            pace_history[dog_id].append((
                race_date,
                r['NormTime'] if pd.notna(r['NormTime']) else None,
                r['Split'] if pd.notna(r['Split']) else None,
                r['Position']
            ))
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    pred_df = pd.DataFrame(predictions)
    pred_df['BSP'] = pd.to_numeric(pred_df['BSP'], errors='coerce')
    pred_df = pred_df.dropna(subset=['BSP'])
    
    print(f"\n[4/5] Total predictions: {len(pred_df):,}")
    
    # Get leaders per race
    print("\n[5/5] Results...")
    print("="*70)
    
    def get_leaders(df, prob_col):
        return df.loc[df.groupby('RaceID')[prob_col].idxmax()]
    
    def backtest(df, label):
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        returns = df[df['Won'] == 1]['BSP'].sum()
        profit = returns - len(df)
        roi = profit / len(df) * 100
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    print("\n--- SINGLE MODELS ---")
    elo_leaders = get_leaders(pred_df, 'EloProb')
    pace_leaders = get_leaders(pred_df, 'PaceProb')
    backtest(elo_leaders[(elo_leaders['BSP'] >= 2) & (elo_leaders['BSP'] <= 10)], "Elo $2-$10")
    backtest(pace_leaders[(pace_leaders['BSP'] >= 2) & (pace_leaders['BSP'] <= 10)], "Pace $2-$10")
    
    print("\n--- COMBINED METHODS ---")
    avg_leaders = get_leaders(pred_df, 'AvgProb')
    max_leaders = get_leaders(pred_df, 'MaxProb')
    backtest(avg_leaders[(avg_leaders['BSP'] >= 2) & (avg_leaders['BSP'] <= 10)], "Average $2-$10")
    backtest(max_leaders[(max_leaders['BSP'] >= 2) & (max_leaders['BSP'] <= 10)], "Max $2-$10")
    
    print("\n--- BOTH AGREE ---")
    # Find races where both models pick the same dog
    elo_picks = elo_leaders[['RaceID', 'GreyhoundID']].rename(columns={'GreyhoundID': 'EloPick'})
    pace_picks = pace_leaders[['RaceID', 'GreyhoundID']].rename(columns={'GreyhoundID': 'PacePick'})
    agree = elo_picks.merge(pace_picks, on='RaceID')
    agree = agree[agree['EloPick'] == agree['PacePick']]
    
    both_agree = pred_df[pred_df['RaceID'].isin(agree['RaceID'])]
    both_agree_leaders = get_leaders(both_agree, 'AvgProb')
    
    backtest(both_agree_leaders[(both_agree_leaders['BSP'] >= 2) & (both_agree_leaders['BSP'] <= 10)], "BOTH AGREE $2-$10")
    backtest(both_agree_leaders[(both_agree_leaders['BSP'] >= 3) & (both_agree_leaders['BSP'] <= 8)], "BOTH AGREE $3-$8")
    
    print("\n--- BY TIER (BOTH AGREE) ---")
    for tier, name in [(2, 'Metro'), (1, 'Provincial'), (0, 'Country')]:
        t = both_agree_leaders[(both_agree_leaders['Tier'] == tier) & 
                               (both_agree_leaders['BSP'] >= 2) & (both_agree_leaders['BSP'] <= 10)]
        if len(t) > 100:
            backtest(t, f"{name}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_combined()
