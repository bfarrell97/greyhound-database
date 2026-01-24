
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = 'greyhound_racing.db'

def analyze_losses():
    print("Loading data for < $2.25 favourites...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        r.Distance,
        t.TrackName,
        ge.Box,
        ge.Position,
        ge.FinishTime,
        ge.Split,
        ge.StartingPrice,
        ge.Weight,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.StartingPrice IS NOT NULL
      AND CAST(REPLACE(REPLACE(ge.StartingPrice, '$', ''), 'F', '') AS FLOAT) < 2.25
      AND ge.Position IS NOT NULL
    """
    
    # Custom price parser to handle '$2.10F' etc
    def parse_price(x):
        try:
            return float(str(x).replace('$', '').replace('F', ''))
        except:
            return None

    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Process Data
    df['Price'] = df['StartingPrice'].apply(parse_price)
    df = df[df['Price'] < 2.25].copy()
    
    # Outcome: 1 = Win, 0 = Loss
    # We want predictors of LOSS, so Outcome=1 means LOST
    df['DidLose'] = df['Position'].apply(lambda x: 0 if str(x) == '1' else 1)
    
    print(f"Total Cohort (Odds < 2.25): {len(df)}")
    print(f"Loss Rate: {df['DidLose'].mean()*100:.1f}%")
    
    # 1. Box Analysis
    print("\n--- Loss Rate by Box ---")
    box_stats = df.groupby('Box')['DidLose'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    box_stats['Loss%'] = box_stats['mean'] * 100
    print(box_stats[['count', 'Loss%']])
    
    # 2. Track Analysis (Min 50 bets)
    print("\n--- High Loss Tracks (Min 50 bets) ---")
    track_stats = df.groupby('TrackName')['DidLose'].agg(['count', 'mean'])
    track_stats = track_stats[track_stats['count'] >= 50].sort_values('mean', ascending=False).head(10)
    track_stats['Loss%'] = track_stats['mean'] * 100
    print(track_stats[['count', 'Loss%']])
    
    # 3. Distance Analysis
    print("\n--- Loss Rate by Distance ---")
    # Bin distances: <350, 350-450, 450-550, 550+
    def bin_dist(d):
        try: d = float(d)
        except: return 'Unknown'
        if d < 350: return '< 350m'
        elif d < 450: return '350-450m'
        elif d < 550: return '450-550m'
        else: return '550m+'
        
    df['DistBin'] = df['Distance'].apply(bin_dist)
    dist_stats = df.groupby('DistBin')['DidLose'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    dist_stats['Loss%'] = dist_stats['mean'] * 100
    print(dist_stats[['count', 'Loss%']])
    
    # 4. Weight Analysis
    # Low weight (< 27kg) vs High Weight (> 33kg)
    print("\n--- Weight Analysis ---")
    def bin_weight(w):
        try: w = float(w)
        except: return None
        if w < 27: return 'Light (<27kg)'
        if w > 33: return 'Heavy (>33kg)'
        return 'Medium'
        
    df['WeightBin'] = df['Weight'].apply(bin_weight)
    w_stats = df.groupby('WeightBin')['DidLose'].agg(['count', 'mean'])
    w_stats['Loss%'] = w_stats['mean'] * 100
    print(w_stats[['count', 'Loss%']])
    
    return

if __name__ == "__main__":
    analyze_losses()
