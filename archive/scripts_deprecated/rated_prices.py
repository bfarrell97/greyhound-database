"""
Rated Prices Model: Elo + Pace Combined
========================================
Develops rated odds from Elo and Pace models, compares to market, finds value.
Train: 2020-2023, Test: 2024-2025
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime
import math

# Track tiers for weighted Elo K-factor
METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_k_factor(track):
    if track in METRO_TRACKS: return 40
    elif track in PROVINCIAL_TRACKS: return 32
    else: return 24

def softmax(values, temp=1.0):
    """Convert values to probabilities via softmax"""
    exp_vals = np.exp(np.array(values) / temp)
    return exp_vals / exp_vals.sum()

def run_rated_prices():
    print("="*70)
    print("RATED PRICES MODEL: ELO + PACE COMBINED")
    print("="*70)
    print("Train: 2020-2023, Test: 2024-2025")
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load pace model
    print("\n[1/5] Loading Pace Model...")
    with open('models/pace_xgb_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    pace_model = artifacts['model']
    
    # Load benchmarks
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    
    # Load ALL data 2020-2025
    print("[2/5] Loading race data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate
    """
    df = pd.read_sql_query(query, conn)
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = df['Position'] == 1
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    print(f"Loaded {len(df):,} entries (2020-2025)")
    
    # Load historical pace data
    print("[3/5] Loading pace history...")
    hist_query = """
    SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < '2024-01-01'
      AND ge.FinishTime IS NOT NULL AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate
    """
    hist_df = pd.read_sql_query(hist_query, conn)
    conn.close()
    
    hist_df['MeetingDate'] = pd.to_datetime(hist_df['MeetingDate'])
    hist_df = hist_df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
    hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['MedianTime']
    hist_df = hist_df.dropna(subset=['NormTime'])
    
    # Calculate rolling features
    hist_df = hist_df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = hist_df.groupby('GreyhoundID')
    hist_df['Lag1'] = g['NormTime'].shift(0)
    hist_df['Lag2'] = g['NormTime'].shift(1)
    hist_df['Lag3'] = g['NormTime'].shift(2)
    hist_df['Roll3'] = g['NormTime'].transform(lambda x: x.rolling(3, min_periods=3).mean())
    hist_df['Roll5'] = g['NormTime'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    hist_df['PrevDate'] = g['MeetingDate'].shift(0)
    
    hist_df = hist_df.dropna(subset=['Roll5'])
    latest = hist_df.groupby('GreyhoundID').last().reset_index()
    feature_lookup = {row['GreyhoundID']: row for _, row in latest.iterrows()}
    
    print(f"Dogs with pace history: {len(feature_lookup):,}")
    
    # [4/5] Process races
    print("[4/5] Processing races...")
    
    elo_ratings = defaultdict(lambda: 1500)
    predictions = []
    races_processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        k = get_k_factor(track)
        
        # Get Elo ratings
        race_elo = {r['GreyhoundID']: elo_ratings[r['GreyhoundID']] 
                    for _, r in race_df.iterrows()}
        
        # Get Pace predictions (lower = faster = better)
        pace_preds = {}
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            if dog_id in feature_lookup:
                feat = feature_lookup[dog_id]
                days_since = min((race_date - feat['PrevDate']).days, 60) if pd.notna(feat['PrevDate']) else 30
                X = np.array([[feat['Lag1'], feat['Lag2'], feat['Lag3'], 
                               feat['Roll3'], feat['Roll5'], days_since, 
                               r['Box'], distance]])
                pace_preds[dog_id] = pace_model.predict(X)[0]
        
        if len(pace_preds) < 4:
            continue
        
        # Get common dogs (have both Elo and Pace)
        common_dogs = set(race_elo.keys()) & set(pace_preds.keys())
        if len(common_dogs) < 4:
            continue
        
        # Calculate probabilities
        elo_vals = [race_elo[d] for d in common_dogs]
        pace_vals = [-pace_preds[d] for d in common_dogs]  # Negate so higher = better
        
        p_elo = softmax(elo_vals, temp=400)  # Same temp as Elo formula
        p_pace = softmax(pace_vals, temp=0.5)  # Adjust temp for pace
        
        # Map back to dog IDs
        dogs = list(common_dogs)
        elo_prob = {d: p for d, p in zip(dogs, p_elo)}
        pace_prob = {d: p for d, p in zip(dogs, p_pace)}
        
        # Store TEST period predictions (2024-2025)
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                if dog_id not in common_dogs:
                    continue
                
                p_e = elo_prob[dog_id]
                p_p = pace_prob[dog_id]
                
                # Combination methods
                p_avg = (p_e + p_p) / 2
                p_geo = math.sqrt(p_e * p_p)
                p_w60 = 0.6 * p_e + 0.4 * p_p  # 60% Elo, 40% Pace
                p_w40 = 0.4 * p_e + 0.6 * p_p  # 40% Elo, 60% Pace
                
                # Renormalize after combination (optional but cleaner)
                
                predictions.append({
                    'RaceID': race_id,
                    'GreyhoundID': dog_id,
                    'Won': r['Won'],
                    'BSP': r['BSP'],
                    'Track': track,
                    'Distance': distance,
                    'P_elo': p_e,
                    'P_pace': p_p,
                    'P_avg': p_avg,
                    'P_geo': p_geo,
                    'P_w60': p_w60,
                    'P_w40': p_w40,
                    'Year': race_date.year
                })
        
        # Update Elo ratings
        total_exp = sum(np.exp(race_elo[d] / 400) for d in race_elo)
        expected = {d: np.exp(race_elo[d] / 400) / total_exp for d in race_elo}
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            actual = 1.0 if r['Won'] else 0.0
            elo_ratings[dog_id] += k * (actual - expected[dog_id])
        
        races_processed += 1
        if races_processed % 20000 == 0:
            print(f"  {races_processed:,} races...")
    
    print(f"  Total: {races_processed:,} races")
    
    # [5/5] Backtest
    print("\n[5/5] Backtest Results...")
    print("="*70)
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.dropna(subset=['BSP'])
    
    # Calculate rated odds and value
    for method in ['P_avg', 'P_geo', 'P_w60', 'P_w40', 'P_elo', 'P_pace']:
        pred_df[f'Odds_{method}'] = 1 / pred_df[method]
        pred_df[f'Value_{method}'] = (pred_df['BSP'] * pred_df[method]) - 1
    
    print(f"\nTotal predictions: {len(pred_df):,}")
    
    # Test value betting at different thresholds
    def test_value(df, prob_col, thresholds=[0.05, 0.10, 0.15, 0.20, 0.25]):
        value_col = f'Value_{prob_col}'
        print(f"\n--- {prob_col} ---")
        print(f"{'Threshold':<12} {'Bets':>8} {'Wins':>8} {'SR%':>8} {'Profit':>10} {'ROI%':>8}")
        print("-"*60)
        
        for thresh in thresholds:
            bets = df[(df[value_col] >= thresh) & (df['BSP'] >= 2) & (df['BSP'] <= 20)]
            if len(bets) < 50:
                print(f"{thresh*100:.0f}%          {len(bets):>8} N/A")
                continue
            
            wins = bets['Won'].sum()
            sr = wins / len(bets) * 100
            returns = bets[bets['Won']]['BSP'].sum()
            profit = returns - len(bets)
            roi = profit / len(bets) * 100
            print(f"{thresh*100:.0f}%          {len(bets):>8} {wins:>8} {sr:>7.1f}% {profit:>9.1f}u {roi:>7.1f}%")
    
    for method in ['P_avg', 'P_geo', 'P_w60', 'P_w40', 'P_elo', 'P_pace']:
        test_value(pred_df, method)
    
    # Best method breakdown by year
    print("\n" + "="*70)
    print("BEST METHOD BY YEAR (10% threshold, $2-$20)")
    print("="*70)
    
    best_method = 'P_avg'
    value_col = f'Value_{best_method}'
    
    for year in [2024, 2025]:
        y = pred_df[(pred_df['Year'] == year) & (pred_df[value_col] >= 0.10) & 
                    (pred_df['BSP'] >= 2) & (pred_df['BSP'] <= 20)]
        if len(y) > 0:
            wins = y['Won'].sum()
            sr = wins / len(y) * 100
            returns = y[y['Won']]['BSP'].sum()
            profit = returns - len(y)
            roi = profit / len(y) * 100
            print(f"{year}: {len(y):,} bets, {wins:,} wins ({sr:.1f}%), {profit:.1f}u, ROI: {roi:+.1f}%")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_rated_prices()
