"""
Metro Elo + Pace Model Combined Test
=====================================
Tests if Metro Elo edge (+5.3%) stacks with Pace model edge.
Strategies:
1. Metro Elo Leader only
2. Metro Pace Leader only  
3. Metro - Both Elo AND Pace agree on leader
4. Metro - Pace Leader in Elo Top 3
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime

METRO_TRACKS = {
    'Wentworth Park', 'Albion Park', 'Angle Park',
    'Sandown Park', 'The Meadows', 'Cannington'
}

K_METRO = 40

def run_test():
    print("="*70)
    print("METRO ELO + PACE MODEL COMBINED")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load pace model
    print("\n[1/4] Loading Pace Model...")
    with open('models/pace_xgb_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    pace_model = artifacts['model']
    
    # Load benchmarks
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    
    # Load 2024-2025 data (only Metro tracks)
    print("[2/4] Loading Metro race data...")
    track_list = "', '".join(METRO_TRACKS)
    query = f"""
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2024-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
      AND t.TrackName IN ('{track_list}')
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
    
    print(f"Loaded {len(df):,} Metro entries")
    
    # Load historical data for pace features
    print("[3/4] Loading pace history...")
    hist_query = """
    SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < '2025-01-01'
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
    
    # Get latest features per dog
    hist_df = hist_df.dropna(subset=['Roll5'])
    latest = hist_df.groupby('GreyhoundID').last().reset_index()
    feature_lookup = {row['GreyhoundID']: row for _, row in latest.iterrows()}
    
    print(f"Dogs with pace history: {len(feature_lookup):,}")
    
    # [4/4] Process races
    print("[4/4] Processing Metro races...")
    
    elo_ratings = defaultdict(lambda: 1500)
    predictions = []
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        
        # Get Elo ratings
        race_elo = {row['GreyhoundID']: elo_ratings[row['GreyhoundID']] 
                    for _, row in race_df.iterrows()}
        
        # Get Pace predictions
        pace_preds = {}
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            if dog_id in feature_lookup:
                feat = feature_lookup[dog_id]
                days_since = min((race_date - feat['PrevDate']).days, 60) if pd.notna(feat['PrevDate']) else 30
                X = np.array([[feat['Lag1'], feat['Lag2'], feat['Lag3'], 
                               feat['Roll3'], feat['Roll5'], days_since, 
                               row['Box'], distance]])
                pace_preds[dog_id] = pace_model.predict(X)[0]
        
        if len(pace_preds) < 4:
            continue
        
        # Elo leader
        elo_sorted = sorted(race_elo.items(), key=lambda x: x[1], reverse=True)
        elo_leader = elo_sorted[0][0]
        elo_top3 = set([x[0] for x in elo_sorted[:3]])
        elo_gap = elo_sorted[0][1] - elo_sorted[1][1]
        
        # Pace leader
        pace_sorted = sorted(pace_preds.items(), key=lambda x: x[1])
        pace_leader = pace_sorted[0][0]
        pace_gap = pace_sorted[1][1] - pace_sorted[0][1]
        
        # Store 2025 predictions
        if race_date >= datetime(2025, 1, 1):
            for _, row in race_df.iterrows():
                dog_id = row['GreyhoundID']
                if dog_id not in pace_preds:
                    continue
                
                predictions.append({
                    'RaceID': race_id,
                    'GreyhoundID': dog_id,
                    'Won': row['Won'],
                    'BSP': row['BSP'],
                    'Distance': distance,
                    'IsEloLeader': dog_id == elo_leader,
                    'IsPaceLeader': dog_id == pace_leader,
                    'BothAgree': (dog_id == elo_leader) and (dog_id == pace_leader),
                    'PaceInEloTop3': (dog_id == pace_leader) and (dog_id in elo_top3),
                    'EloGap': elo_gap if dog_id == elo_leader else 0,
                    'PaceGap': pace_gap if dog_id == pace_leader else 0
                })
        
        # Update Elo
        total_exp = sum(np.exp(r / 400) for r in race_elo.values())
        expected = {d: np.exp(race_elo[d] / 400) / total_exp for d in race_elo}
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            actual = 1.0 if row['Won'] else 0.0
            elo_ratings[dog_id] += K_METRO * (actual - expected[dog_id])
    
    print(f"Predictions for 2025: {len(predictions):,}")
    
    # Results
    print("\n" + "="*70)
    print("METRO RESULTS (2025 BSP $3-$8)")
    print("="*70)
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.dropna(subset=['BSP'])
    
    def calc_roi(subset, label):
        f = subset[(subset['BSP'] >= 3) & (subset['BSP'] <= 8)]
        if len(f) < 30:
            print(f"{label}: N/A (n={len(f)})")
            return None
        wins = f['Won'].sum()
        sr = wins / len(f) * 100
        returns = f[f['Won']]['BSP'].sum()
        profit = returns - len(f)
        roi = profit / len(f) * 100
        print(f"{label}: {len(f):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
        return roi
    
    print("\n--- SINGLE MODEL ---")
    calc_roi(pred_df[pred_df['IsEloLeader']], "Elo Leader")
    calc_roi(pred_df[pred_df['IsPaceLeader']], "Pace Leader")
    
    print("\n--- COMBINED ---")
    calc_roi(pred_df[pred_df['BothAgree']], "BOTH AGREE (Elo + Pace)")
    calc_roi(pred_df[pred_df['PaceInEloTop3']], "Pace Leader in Elo Top 3")
    
    # With gaps
    print("\n--- WITH GAP FILTERS ---")
    both = pred_df[pred_df['BothAgree']]
    both_gap = both[(both['EloGap'] >= 50) | (both['PaceGap'] >= 0.15)]
    calc_roi(both_gap, "BOTH + (EloGap>=50 OR PaceGap>=0.15)")
    
    pace_top3 = pred_df[pred_df['PaceInEloTop3']]
    pace_gap = pace_top3[pace_top3['PaceGap'] >= 0.15]
    calc_roi(pace_gap, "Pace+EloTop3 + PaceGap>=0.15")
    
    # Mid distance
    print("\n--- MID DISTANCE (400-550m) ---")
    mid = pred_df[(pred_df['Distance'] >= 400) & (pred_df['Distance'] < 550)]
    calc_roi(mid[mid['IsEloLeader']], "Elo + Mid")
    calc_roi(mid[mid['IsPaceLeader']], "Pace + Mid")
    calc_roi(mid[mid['BothAgree']], "BOTH + Mid")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_test()
