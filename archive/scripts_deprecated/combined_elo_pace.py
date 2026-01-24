"""
Combined Elo + Pace Strategy Test
==================================
Tests multiple combinations:
1. BOTH AGREE: Elo leader = Pace leader
2. PACE + ELO FILTER: Pace leader is in Elo top 3
3. ELO + PACE FILTER: Elo leader is in Pace top 3
4. WEIGHTED: Average rank of Elo and Pace
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime

def run_combined_test():
    print("="*70)
    print("COMBINED ELO + PACE STRATEGY TEST")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load pace model
    print("\n[1/5] Loading Pace Model...")
    with open('models/pace_xgb_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    pace_model = artifacts['model']
    
    # Load benchmarks
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    
    # Load 2024-2025 data
    print("[2/5] Loading race data...")
    query = """
    SELECT ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           r.Distance, rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2024-01-01' AND '2025-11-30'
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
    
    print(f"Loaded {len(df):,} entries")
    
    # Load historical pace data for features
    print("[3/5] Loading historical pace data...")
    hist_query = """
    SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < '2025-12-01'
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
    
    # Build lookup by (GreyhoundID, Date) -> features
    hist_df = hist_df.dropna(subset=['Roll5'])
    latest_features = hist_df.groupby('GreyhoundID').last().reset_index()
    feature_lookup = {row['GreyhoundID']: row for _, row in latest_features.iterrows()}
    
    print(f"Dogs with pace history: {len(feature_lookup):,}")
    
    # [4/5] Process races and calculate both Elo and Pace rankings
    print("[4/5] Calculating Elo and Pace rankings...")
    
    K = 32
    elo_ratings = defaultdict(lambda: 1500)
    predictions = []
    races_processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:  # Need at least 4 dogs
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        
        # Get Elo ratings for each dog
        elo_ranks = {}
        for _, row in race_df.iterrows():
            elo_ranks[row['GreyhoundID']] = elo_ratings[row['GreyhoundID']]
        
        # Get Pace predictions
        pace_preds = {}
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            if dog_id in feature_lookup:
                feat = feature_lookup[dog_id]
                days_since = (race_date - feat['PrevDate']).days if pd.notna(feat['PrevDate']) else 30
                days_since = min(days_since, 60)
                X = np.array([[feat['Lag1'], feat['Lag2'], feat['Lag3'], 
                               feat['Roll3'], feat['Roll5'], days_since, 
                               row['Box'], distance]])
                pace_preds[dog_id] = pace_model.predict(X)[0]
        
        if len(pace_preds) < 4:
            continue
        
        # Rank by Elo (higher is better)
        elo_sorted = sorted(elo_ranks.items(), key=lambda x: x[1], reverse=True)
        elo_leader = elo_sorted[0][0]
        elo_top3 = set([x[0] for x in elo_sorted[:3]])
        elo_gap = elo_sorted[0][1] - elo_sorted[1][1] if len(elo_sorted) >= 2 else 0
        
        # Rank by Pace (lower is better)
        pace_sorted = sorted(pace_preds.items(), key=lambda x: x[1])
        pace_leader = pace_sorted[0][0]
        pace_top3 = set([x[0] for x in pace_sorted[:3]])
        pace_gap = pace_sorted[1][1] - pace_sorted[0][1] if len(pace_sorted) >= 2 else 0
        
        # Combined rank (average of elo rank and pace rank)
        all_dogs = set(elo_ranks.keys()) & set(pace_preds.keys())
        elo_rank_map = {d: i+1 for i, (d, _) in enumerate(elo_sorted) if d in all_dogs}
        pace_rank_map = {d: i+1 for i, (d, _) in enumerate(pace_sorted) if d in all_dogs}
        
        combined_ranks = {d: (elo_rank_map.get(d, 8) + pace_rank_map.get(d, 8)) / 2 
                          for d in all_dogs}
        combo_leader = min(combined_ranks, key=combined_ranks.get)
        
        # Store predictions for 2025 test set
        if race_date >= datetime(2025, 1, 1):
            for _, row in race_df.iterrows():
                dog_id = row['GreyhoundID']
                if dog_id not in all_dogs:
                    continue
                    
                predictions.append({
                    'RaceID': race_id,
                    'GreyhoundID': dog_id,
                    'Won': row['Won'],
                    'BSP': row['BSP'],
                    'Distance': distance,
                    'EloLeader': dog_id == elo_leader,
                    'PaceLeader': dog_id == pace_leader,
                    'ComboLeader': dog_id == combo_leader,
                    'BothAgree': (dog_id == elo_leader) and (dog_id == pace_leader),
                    'EloLeaderInPaceTop3': (dog_id == elo_leader) and (dog_id in pace_top3),
                    'PaceLeaderInEloTop3': (dog_id == pace_leader) and (dog_id in elo_top3),
                    'EloGap': elo_gap if dog_id == elo_leader else 0,
                    'PaceGap': pace_gap if dog_id == pace_leader else 0,
                    'ComboRank': combined_ranks.get(dog_id, 8)
                })
        
        # Update Elo ratings
        total_exp = sum(np.exp(r / 400) for r in elo_ranks.values())
        expected = {d: np.exp(elo_ranks[d] / 400) / total_exp for d in elo_ranks}
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            actual = 1.0 if row['Won'] else 0.0
            elo_ratings[dog_id] += K * (actual - expected[dog_id])
        
        races_processed += 1
        if races_processed % 5000 == 0:
            print(f"  {races_processed:,} races...")
    
    print(f"  Total: {races_processed:,} races")
    
    # [5/5] Analyze results
    print("\n[5/5] Results...")
    print("="*70)
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.dropna(subset=['BSP'])
    
    # Filter functions
    def calc_roi(subset, label):
        if len(subset) < 50:
            print(f"{label}: N/A (n={len(subset)})")
            return None
        
        f = subset[(subset['BSP'] >= 3) & (subset['BSP'] <= 8)]
        if len(f) < 30:
            print(f"{label}: N/A after $3-$8 filter (n={len(f)})")
            return None
        
        wins = f['Won'].sum()
        sr = wins / len(f) * 100
        returns = f[f['Won']]['BSP'].sum()
        profit = returns - len(f)
        roi = profit / len(f) * 100
        print(f"{label}: {len(f):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
        return roi
    
    print("\n--- STRATEGY COMPARISON (@$3-$8) ---")
    print("-"*60)
    
    # 1. Elo Leader only
    calc_roi(pred_df[pred_df['EloLeader']], "Elo Leader")
    
    # 2. Pace Leader only
    calc_roi(pred_df[pred_df['PaceLeader']], "Pace Leader")
    
    # 3. Combo Leader (avg rank)
    calc_roi(pred_df[pred_df['ComboLeader']], "Combo Leader (Avg Rank)")
    
    # 4. Both Agree
    calc_roi(pred_df[pred_df['BothAgree']], "BOTH AGREE")
    
    # 5. Elo Leader in Pace Top 3
    calc_roi(pred_df[pred_df['EloLeaderInPaceTop3']], "Elo Leader + Pace Top 3")
    
    # 6. Pace Leader in Elo Top 3
    calc_roi(pred_df[pred_df['PaceLeaderInEloTop3']], "Pace Leader + Elo Top 3")
    
    # With Middle Distance filter
    print("\n--- WITH MIDDLE DISTANCE (400-550m) ---")
    print("-"*60)
    
    mid = pred_df[(pred_df['Distance'] >= 400) & (pred_df['Distance'] < 550)]
    
    calc_roi(mid[mid['EloLeader']], "Elo Leader + Mid Dist")
    calc_roi(mid[mid['PaceLeader']], "Pace Leader + Mid Dist")
    calc_roi(mid[mid['ComboLeader']], "Combo Leader + Mid Dist")
    calc_roi(mid[mid['BothAgree']], "BOTH AGREE + Mid Dist")
    calc_roi(mid[mid['PaceLeaderInEloTop3']], "Pace+EloTop3 + Mid Dist")
    
    # With Gap filters
    print("\n--- WITH GAP FILTERS ---")
    print("-"*60)
    
    both = pred_df[pred_df['BothAgree']]
    both_gap = both[(both['EloGap'] >= 50) | (both['PaceGap'] >= 0.15)]
    calc_roi(both_gap, "BOTH AGREE + (EloGap>=50 OR PaceGap>=0.15)")
    
    mid_both = mid[mid['BothAgree']]
    mid_both_gap = mid_both[(mid_both['EloGap'] >= 50) | (mid_both['PaceGap'] >= 0.15)]
    calc_roi(mid_both_gap, "BOTH AGREE + Mid + Gap")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_combined_test()
