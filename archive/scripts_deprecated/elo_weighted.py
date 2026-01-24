"""
Track-Weighted Elo Model
========================
Elo ratings with different K-factors based on track tier:
- Metro: K=40 (wins at metro tracks worth more)
- Provincial: K=32 (standard)
- Country: K=24 (wins at country tracks worth less)

Train: 2024, Test: 2025
"""
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

# Track Tier Definitions
METRO_TRACKS = {
    'Wentworth Park', 'Albion Park', 'Angle Park',
    'Sandown Park', 'The Meadows', 'Cannington'
}

PROVINCIAL_TRACKS = {
    'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli',
    'Dapto', 'Maitland', 'Goulburn', 'Ipswich', 'Q Straight',
    'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
    'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'
}

# K-factors by tier
K_METRO = 40      # Wins at metro worth more
K_PROVINCIAL = 32  # Standard
K_COUNTRY = 24     # Wins at country worth less

def get_track_tier(track_name):
    """Get tier for track"""
    if track_name in METRO_TRACKS:
        return 'Metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'Provincial'
    else:
        return 'Country'

def get_k_factor(track_name):
    """Get K-factor for track"""
    if track_name in METRO_TRACKS:
        return K_METRO
    elif track_name in PROVINCIAL_TRACKS:
        return K_PROVINCIAL
    else:
        return K_COUNTRY

def run_elo():
    print("="*70)
    print("TRACK-WEIGHTED ELO MODEL")
    print("="*70)
    print(f"K-factors: Metro={K_METRO}, Provincial={K_PROVINCIAL}, Country={K_COUNTRY}")
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load 2024-2025 data
    print("\n[1/3] Loading race data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, r.Distance,
           rm.MeetingDate, t.TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2024-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = df['Position'] == 1
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_track_tier)
    
    print(f"Loaded {len(df):,} entries")
    print(f"\nTrack Distribution:")
    print(f"  Metro:      {(df['Tier']=='Metro').sum():,} ({(df['Tier']=='Metro').mean()*100:.1f}%)")
    print(f"  Provincial: {(df['Tier']=='Provincial').sum():,} ({(df['Tier']=='Provincial').mean()*100:.1f}%)")
    print(f"  Country:    {(df['Tier']=='Country').sum():,} ({(df['Tier']=='Country').mean()*100:.1f}%)")
    
    # [2/3] Calculate Elo ratings
    print("\n[2/3] Calculating Elo ratings...")
    
    ratings = defaultdict(lambda: 1500)
    predictions = []
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        track = race_df['TrackName'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        k = get_k_factor(track)
        
        # Get current ratings
        race_ratings = {row['GreyhoundID']: ratings[row['GreyhoundID']] 
                        for _, row in race_df.iterrows()}
        
        # Expected (softmax)
        total_exp = sum(np.exp(r / 400) for r in race_ratings.values())
        expected = {d: np.exp(race_ratings[d] / 400) / total_exp for d in race_ratings}
        
        # Leader and gap
        elo_sorted = sorted(race_ratings.items(), key=lambda x: x[1], reverse=True)
        elo_leader = elo_sorted[0][0]
        elo_gap = elo_sorted[0][1] - elo_sorted[1][1] if len(elo_sorted) >= 2 else 0
        
        # Store 2025 predictions
        if race_date >= datetime(2025, 1, 1):
            for _, row in race_df.iterrows():
                dog_id = row['GreyhoundID']
                if dog_id == elo_leader:
                    predictions.append({
                        'RaceID': race_id,
                        'Rating': race_ratings[dog_id],
                        'Gap': elo_gap,
                        'Won': row['Won'],
                        'BSP': row['BSP'],
                        'Distance': distance,
                        'Tier': tier
                    })
        
        # Update ratings with track-weighted K
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            actual = 1.0 if row['Won'] else 0.0
            ratings[dog_id] += k * (actual - expected[dog_id])
    
    print(f"  Predictions for 2025: {len(predictions):,}")
    
    # [3/3] Results
    print("\n[3/3] Results...")
    print("="*70)
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.dropna(subset=['BSP'])
    
    def calc_roi(subset, label):
        f = subset[(subset['BSP'] >= 3) & (subset['BSP'] <= 8)]
        if len(f) < 50:
            print(f"{label}: N/A (n<50)")
            return
        wins = f['Won'].sum()
        sr = wins / len(f) * 100
        returns = f[f['Won']]['BSP'].sum()
        profit = returns - len(f)
        roi = profit / len(f) * 100
        print(f"{label}: {len(f):,} bets, {wins:,} wins ({sr:.1f}%), Profit: {profit:.1f}u, ROI: {roi:+.1f}%")
        return roi
    
    print("\n--- ALL TRACKS ---")
    calc_roi(pred_df, "Elo Leader @ $3-$8")
    calc_roi(pred_df[pred_df['Gap'] >= 50], "Elo Leader + Gap>=50")
    
    # By mid distance
    mid = pred_df[(pred_df['Distance'] >= 400) & (pred_df['Distance'] < 550)]
    calc_roi(mid, "Elo + Mid Dist")
    calc_roi(mid[mid['Gap'] >= 50], "Elo + Mid + Gap>=50")
    
    print("\n--- BY TRACK TIER ---")
    for tier in ['Metro', 'Provincial', 'Country']:
        t_df = pred_df[pred_df['Tier'] == tier]
        calc_roi(t_df, f"{tier}")
    
    print("\n--- METRO ONLY ---")
    metro = pred_df[pred_df['Tier'] == 'Metro']
    calc_roi(metro, "Metro All")
    calc_roi(metro[metro['Gap'] >= 50], "Metro + Gap>=50")
    metro_mid = metro[(metro['Distance'] >= 400) & (metro['Distance'] < 550)]
    calc_roi(metro_mid, "Metro + Mid Dist")
    calc_roi(metro_mid[metro_mid['Gap'] >= 50], "Metro + Mid + Gap>=50")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_elo()
