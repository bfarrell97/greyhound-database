import pandas as pd
import numpy as np
import sqlite3
import os
import sys

def analyze_steamer_value():
    print("Loading 2025 Data for Steamer Value Analysis...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.Position, ge.BSP, ge.Price5Min, 
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Total Rows: {len(df)}")
    
    # 1. Define Win
    df['Win'] = (pd.to_numeric(df['Position'], errors='coerce') == 1).astype(int)
    
    # 2. Define Market Move
    # Movement is (5Min / BSP)
    # > 1.0 means it STEAMED (Price dropped from 5min to BSP)
    # < 1.0 means it DRIFTED
    df['MoveRatio'] = df['Price5Min'] / df['BSP']
    
    # 3. Segments
    segments = [
        ('All Dogs', df),
        ('Steamers (Ratio > 1.05)', df[df['MoveRatio'] > 1.05]),
        ('Heavy Steamers (Ratio > 1.20)', df[df['MoveRatio'] > 1.20]),
        ('Extreme Steamers (Ratio > 1.50)', df[df['MoveRatio'] > 1.50]),
        ('Static (0.95 - 1.05)', df[(df['MoveRatio'] >= 0.95) & (df['MoveRatio'] <= 1.05)]),
        ('Drifters (Ratio < 0.95)', df[df['MoveRatio'] < 0.95])
    ]
    
    print("\n" + "="*80)
    print(f"{'Segment':<30} | {'Count':<8} | {'Win%':<6} | {'Early ROI':<10} | {'BSP ROI':<8}")
    print("-" * 80)
    
    results = []
    for name, sub in segments:
        if len(sub) == 0: continue
        
        count = len(sub)
        win_rate = sub['Win'].mean() * 100
        
        # Calculate ROI at 5Min Price (Early)
        # Assuming $10 flat stakes, 8% commission on wins
        ret_early = np.where(sub['Win']==1, 10*(sub['Price5Min']-1)*0.92, -10).sum()
        roi_early = ret_early / (count*10) * 100
        
        # Calculate Lay ROI at 5Min Price
        # Stake $10, 8% commission on net profit
        # If dog loses (Win=0): Profit = Stake * (1 - Comm)
        # If dog wins (Win=1): Loss = Stake * (Price - 1)
        lay_ret_early = np.where(sub['Win']==0, 10*0.92, -10*(sub['Price5Min']-1)).sum()
        lay_roi_early = lay_ret_early / (count*10) * 100

        # Calculate ROI at BSP
        ret_bsp = np.where(sub['Win']==1, 10*(sub['BSP']-1)*0.92, -10).sum()
        roi_bsp = ret_bsp / (count*10) * 100
        
        # Calculate Lay ROI at BSP
        lay_ret_bsp = np.where(sub['Win']==0, 10*0.92, -10*(sub['BSP']-1)).sum()
        lay_roi_bsp = lay_ret_bsp / (count*10) * 100

        print(f"{name:<30} | {count:<8} | {win_rate:>5.1f}% | {roi_early:>+8.1f}%   | {lay_roi_early:>+8.1f}%")
        results.append({
            'Segment': name,
            'Count': count,
            'WinRate': win_rate,
            'ROI_Early': roi_early,
            'Lay_ROI_Early': lay_roi_early,
            'ROI_BSP': roi_bsp,
            'Lay_ROI_BSP': lay_roi_bsp,
            'Alpha': roi_early - roi_bsp
        })

    print("="*80)
    
    # Save to CSV for reliability
    res_df = pd.DataFrame(results)
    res_df.to_csv('steamer_analysis.csv', index=False)
    print("\nResults saved to steamer_analysis.csv")

    print("\nInterpretation:")
    print("If Early ROI is positive, the user's hypothesis holds.")
    print("If Early ROI is negative but BSP ROI is MUCH more negative, then Steamers have Value relative to market.")

if __name__ == "__main__":
    analyze_steamer_value()
