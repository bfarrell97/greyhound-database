"""
BSP vs 5-Min Price Comparison Analysis
======================================
Compares prices at 5 minutes before vs BSP to identify edge opportunities.
Period: March 2025 - November 2025
"""
import sqlite3
import pandas as pd
import numpy as np

def analyze():
    print("="*70)
    print("BSP vs 5-MIN PRICE COMPARISON (Mar-Nov 2025)")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load data with both prices
    query = """
    SELECT ge.EntryID, ge.Position, ge.StartingPrice, ge.BSP, ge.Price5Min,
           rm.MeetingDate, t.TrackName, r.RaceNumber, r.Distance
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2025-03-01' AND '2025-11-30'
      AND ge.BSP IS NOT NULL
      AND ge.Price5Min IS NOT NULL
      AND ge.Position IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nLoaded {len(df):,} entries with both BSP and P5 prices")
    
    # Clean data
    df['Won'] = df['Position'].apply(lambda x: str(x).strip() == '1')
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')
    df['SP'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df = df.dropna(subset=['BSP', 'Price5Min'])
    
    print(f"After cleaning: {len(df):,} entries")
    
    # Calculate price difference
    df['PriceDiff'] = df['BSP'] - df['Price5Min']  # Positive = BSP higher than P5
    df['PriceDiffPct'] = (df['PriceDiff'] / df['Price5Min']) * 100
    
    # 1. Overall Price Comparison
    print("\n" + "="*70)
    print("1. OVERALL PRICE COMPARISON")
    print("="*70)
    
    avg_bsp = df['BSP'].mean()
    avg_p5 = df['Price5Min'].mean()
    avg_diff = df['PriceDiff'].mean()
    
    print(f"Average BSP: ${avg_bsp:.2f}")
    print(f"Average P5:  ${avg_p5:.2f}")
    print(f"Average Difference: ${avg_diff:.2f} ({avg_diff/avg_p5*100:.1f}%)")
    
    # 2. Winners vs Losers
    print("\n" + "="*70)
    print("2. WINNERS vs LOSERS PRICE MOVEMENT")
    print("="*70)
    
    winners = df[df['Won']]
    losers = df[~df['Won']]
    
    print(f"\nWinners (n={len(winners):,}):")
    print(f"  Avg BSP:  ${winners['BSP'].mean():.2f}")
    print(f"  Avg P5:   ${winners['Price5Min'].mean():.2f}")
    print(f"  Avg Diff: ${winners['PriceDiff'].mean():.2f} ({winners['PriceDiffPct'].mean():.1f}%)")
    
    print(f"\nLosers (n={len(losers):,}):")
    print(f"  Avg BSP:  ${losers['BSP'].mean():.2f}")
    print(f"  Avg P5:   ${losers['Price5Min'].mean():.2f}")
    print(f"  Avg Diff: ${losers['PriceDiff'].mean():.2f} ({losers['PriceDiffPct'].mean():.1f}%)")
    
    # 3. Edge Detection: P5 vs BSP ROI by odds band
    print("\n" + "="*70)
    print("3. ROI COMPARISON BY ODDS BAND (Backing ALL runners)")
    print("="*70)
    
    bands = [(1.5, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 30)]
    
    print(f"\n{'Odds Band':<12} {'Count':>8} {'Win%':>8} {'P5 ROI':>10} {'BSP ROI':>10} {'Edge':>8}")
    print("-"*60)
    
    for low, high in bands:
        band_p5 = df[(df['Price5Min'] >= low) & (df['Price5Min'] < high)]
        band_bsp = df[(df['BSP'] >= low) & (df['BSP'] < high)]
        
        if len(band_p5) < 100:
            continue
        
        # P5 ROI
        p5_stakes = len(band_p5)
        p5_returns = band_p5[band_p5['Won']]['Price5Min'].sum()
        p5_roi = ((p5_returns - p5_stakes) / p5_stakes) * 100
        
        # BSP ROI
        bsp_stakes = len(band_bsp)
        bsp_returns = band_bsp[band_bsp['Won']]['BSP'].sum()
        bsp_roi = ((bsp_returns - bsp_stakes) / bsp_stakes) * 100 if bsp_stakes > 0 else 0
        
        win_rate = (band_p5['Won'].sum() / len(band_p5)) * 100
        edge = p5_roi - bsp_roi
        
        print(f"${low:>4.1f}-${high:<4.1f} {len(band_p5):>8,} {win_rate:>7.1f}% {p5_roi:>9.1f}% {bsp_roi:>9.1f}% {edge:>+7.1f}%")
    
    # 4. Steam Move Detection
    print("\n" + "="*70)
    print("4. STEAM MOVE ANALYSIS (Price shortened from P5 to BSP)")
    print("="*70)
    
    # Steamer = price shortened (BSP < P5)
    df['Steamer'] = df['BSP'] < df['Price5Min']
    df['Drifter'] = df['BSP'] > df['Price5Min']
    
    steamers = df[df['Steamer'] & (df['Price5Min'] >= 3) & (df['Price5Min'] <= 15)]
    drifters = df[df['Drifter'] & (df['Price5Min'] >= 3) & (df['Price5Min'] <= 15)]
    
    print(f"\nSteamers (price shortened, $3-$15 @ P5): {len(steamers):,}")
    if len(steamers) > 100:
        steam_wins = steamers['Won'].sum()
        steam_sr = (steam_wins / len(steamers)) * 100
        steam_p5_returns = steamers[steamers['Won']]['Price5Min'].sum()
        steam_bsp_returns = steamers[steamers['Won']]['BSP'].sum()
        steam_p5_roi = ((steam_p5_returns - len(steamers)) / len(steamers)) * 100
        steam_bsp_roi = ((steam_bsp_returns - len(steamers)) / len(steamers)) * 100
        print(f"  Win Rate: {steam_sr:.1f}%")
        print(f"  P5 ROI:   {steam_p5_roi:+.1f}%")
        print(f"  BSP ROI:  {steam_bsp_roi:+.1f}%")
    
    print(f"\nDrifters (price lengthened, $3-$15 @ P5): {len(drifters):,}")
    if len(drifters) > 100:
        drift_wins = drifters['Won'].sum()
        drift_sr = (drift_wins / len(drifters)) * 100
        drift_p5_returns = drifters[drifters['Won']]['Price5Min'].sum()
        drift_bsp_returns = drifters[drifters['Won']]['BSP'].sum()
        drift_p5_roi = ((drift_p5_returns - len(drifters)) / len(drifters)) * 100
        drift_bsp_roi = ((drift_bsp_returns - len(drifters)) / len(drifters)) * 100
        print(f"  Win Rate: {drift_sr:.1f}%")
        print(f"  P5 ROI:   {drift_p5_roi:+.1f}%")
        print(f"  BSP ROI:  {drift_bsp_roi:+.1f}%")
    
    # 5. Optimal Time to Bet
    print("\n" + "="*70)
    print("5. CONCLUSION: OPTIMAL BETTING TIME")
    print("="*70)
    
    # Compare overall ROI
    total_bets = len(df[(df['Price5Min'] >= 3) & (df['Price5Min'] <= 8)])
    p5_subset = df[(df['Price5Min'] >= 3) & (df['Price5Min'] <= 8)]
    bsp_subset = df[(df['BSP'] >= 3) & (df['BSP'] <= 8)]
    
    if len(p5_subset) > 0:
        p5_returns = p5_subset[p5_subset['Won']]['Price5Min'].sum()
        p5_roi = ((p5_returns - len(p5_subset)) / len(p5_subset)) * 100
        print(f"\nP5 ($3-$8):  {len(p5_subset):,} bets, ROI = {p5_roi:+.1f}%")
    
    if len(bsp_subset) > 0:
        bsp_returns = bsp_subset[bsp_subset['Won']]['BSP'].sum()
        bsp_roi = ((bsp_returns - len(bsp_subset)) / len(bsp_subset)) * 100
        print(f"BSP ($3-$8): {len(bsp_subset):,} bets, ROI = {bsp_roi:+.1f}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze()
