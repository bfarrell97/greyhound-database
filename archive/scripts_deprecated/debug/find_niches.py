"""
The Niche Hunter (Grid Search)
Goal: Find persistent profitable combinations (Track/Dist/Box or Track/Trainer).
Logic: Must be profitable in BOTH 2024 and 2025 with >20 bets each year.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def hunt_niches():
    print("Loading Data (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.TrainerID,
        tr.FirstName || ' ' || tr.LastName as TrainerName,
        rm.MeetingDate,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    WHERE rm.MeetingDate >= '2024-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    df['Year'] = df['MeetingDate'].dt.year
    
    # ---------------------------------------------------------
    # PART 1: Track x Distance x Box
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("GRID SEARCH: Track x Distance x Box")
    print("="*80)
    
    # Group by Segment and Year
    groups = df.groupby(['TrackName', 'Distance', 'Box', 'Year'])
    
    stats = groups.agg(
        Bets=('IsWin', 'count'),
        Wins=('IsWin', 'sum'),
        Profit=('Odds', lambda x: (x[df.loc[x.index, 'IsWin']==1]-1).sum() - (len(x)-df.loc[x.index, 'IsWin'].sum()))
    ).reset_index()
    
    stats['ROI'] = stats['Profit'] / stats['Bets'] * 100
    
    # pivot to have 2024 and 2025 columns
    pivot = stats.pivot(index=['TrackName', 'Distance', 'Box'], columns='Year', values=['Bets', 'ROI', 'Profit'])
    
    # Flatten columns
    pivot.columns = [f'{c[0]}_{c[1]}' for c in pivot.columns]
    
    # Filter: Profitable in BOTH years, > 20 bets in EACH
    # Note: Adjust column names based on actual years present (2024, 2025)
    if 'ROI_2024' in pivot.columns and 'ROI_2025' in pivot.columns:
        niches = pivot[
            (pivot['ROI_2024'] > 5) & 
            (pivot['ROI_2025'] > 0) & 
            (pivot['Bets_2024'] >= 20) & 
            (pivot['Bets_2025'] >= 20)
        ].copy()
        
        niches['TotalProfit'] = niches['Profit_2024'] + niches['Profit_2025']
        niches = niches.sort_values('TotalProfit', ascending=False)
        
        print(f"\nFound {len(niches)} Persistent Niches (Box Bias):")
        print(niches[['ROI_2024', 'Bets_2024', 'ROI_2025', 'Bets_2025', 'TotalProfit']].to_string())
    else:
        print("Not enough data years for comparison.")
        
    # ---------------------------------------------------------
    # PART 2: Track x Trainer
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("GRID SEARCH: Track x Trainer")
    print("="*80)
    
    trainer_groups = df.groupby(['TrackName', 'TrainerName', 'Year'])
    
    t_stats = trainer_groups.agg(
        Bets=('IsWin', 'count'),
        Wins=('IsWin', 'sum'),
        Profit=('Odds', lambda x: (x[df.loc[x.index, 'IsWin']==1]-1).sum() - (len(x)-df.loc[x.index, 'IsWin'].sum()))
    ).reset_index()
    
    t_stats['ROI'] = t_stats['Profit'] / t_stats['Bets'] * 100
    
    t_pivot = t_stats.pivot(index=['TrackName', 'TrainerName'], columns='Year', values=['Bets', 'ROI', 'Profit'])
    t_pivot.columns = [f'{c[0]}_{c[1]}' for c in t_pivot.columns]
    
    if 'ROI_2024' in t_pivot.columns and 'ROI_2025' in t_pivot.columns:
        t_niches = t_pivot[
            (t_pivot['ROI_2024'] > 10) & 
            (t_pivot['ROI_2025'] > 0) & 
            (t_pivot['Bets_2024'] >= 15) & 
            (t_pivot['Bets_2025'] >= 10)
        ].copy()
        
        t_niches['TotalProfit'] = t_niches['Profit_2024'] + t_niches['Profit_2025']
        t_niches = t_niches.sort_values('TotalProfit', ascending=False)
        
        print(f"\nFound {len(t_niches)} Persistent Niches (Trainer Specialists):")
        print(t_niches[['ROI_2024', 'Bets_2024', 'ROI_2025', 'Bets_2025', 'TotalProfit']].to_string())


if __name__ == "__main__":
    hunt_niches()
