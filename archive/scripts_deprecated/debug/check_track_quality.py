"""
Check Track Quality (Data Completeness Audit)
Goal: Identify tracks where Loser data is missing (Survivor Bias).
Output: 'tier1_tracks.csv' - List of safe tracks to use for modeling.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def check_quality():
    print("Auditing Data Quality (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
    
    # Get all finishes from recent years
    query = """
    SELECT
        t.TrackName,
        r.Distance,
        ge.Position,
        ge.Split,
        ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['IsWin'] = (df['Position'] == '1')
    df['HasSplit'] = df['Split'].notna() & (df['Split'] != '')
    
    # Group by Track
    stats = []
    tracks = df['TrackName'].unique()
    
    print(f"\n{'Track':<20} {'LoserSplit%':<12} {'WinnerSplit%':<12} {'BiasGap':<10} {'Status':<10}")
    print("-" * 70)
    
    for track in tracks:
        t_df = df[df['TrackName'] == track]
        if len(t_df) < 100: continue # Skip tiny tracks
        
        # Calculate coverage
        winners = t_df[t_df['IsWin']]
        losers = t_df[~t_df['IsWin']]
        
        if len(winners) == 0 or len(losers) == 0: continue
        
        win_cov = winners['HasSplit'].mean() * 100
        lose_cov = losers['HasSplit'].mean() * 100
        gap = win_cov - lose_cov
        
        # Classification
        if lose_cov >= 90:
            status = "SAFE"
        elif lose_cov >= 50:
            status = "RISKY"
        else:
            status = "BROKEN"
            
        print(f"{track:<20} {lose_cov:<12.1f} {win_cov:<12.1f} {gap:<10.1f} {status:<10}")
        
        stats.append({
            'Track': track,
            'LoserSplitPct': lose_cov,
            'WinnerSplitPct': win_cov,
            'BiasGap': gap,
            'Status': status,
            'TotalRows': len(t_df)
        })
        
    # Save Tier 1 list
    stats_df = pd.DataFrame(stats)
    tier1 = stats_df[stats_df['Status'] == 'SAFE']['Track'].tolist()
    
    print(f"\nIdentified {len(tier1)} SAFE Tracks: {tier1}")
    
    # Export for other scripts to use
    with open('tier1_tracks.txt', 'w') as f:
        for t in tier1:
            f.write(f"{t}\n")

if __name__ == "__main__":
    check_quality()
