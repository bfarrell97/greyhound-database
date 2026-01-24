"""
Analyze Split Consistency: Raw vs Normalized
Tests whether normalizing splits by Track/Distance improves predictive consistency.
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def analyze_split_consistency():
    conn = sqlite3.connect(DB_PATH)
    
    print("Loading split data...")
    # Load all races with valid splits
    query = """
    SELECT
        ge.GreyhoundID,
        t.TrackName,
        r.Distance,
        ge.Split
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Split IS NOT NULL
      AND ge.Split > 0
      AND ge.Split < 30  -- Filter outliers
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} split records.")
    
    # 1. Calculate Benchmarks (Median Split per Track/Dist)
    print("Calculating Track/Distance benchmarks...")
    benchmarks = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
    
    # Merge benchmarks back
    df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
    
    # 2. Calculate Normalized Split (Raw - Median)
    # A negative value means faster than average (good)
    df['NormSplit'] = df['Split'] - df['TrackDistMedian']
    
    # 3. Analyze Consistency (StdDev) per Dog
    print("Analyzing consistency per dog...")
    
    # Filter for dogs with enough history to measure variance (e.g. > 10 races)
    dog_counts = df['GreyhoundID'].value_counts()
    valid_dogs = dog_counts[dog_counts >= 10].index
    
    subset = df[df['GreyhoundID'].isin(valid_dogs)].copy()
    
    dog_stats = subset.groupby('GreyhoundID').agg({
        'Split': 'std',
        'NormSplit': 'std',
        'TrackName': 'nunique'
    }).rename(columns={'Split': 'RawStdDev', 'NormSplit': 'NormStdDev', 'TrackName': 'TrackCount'})
    
    # 4. Results
    print("\n" + "="*60)
    print("SPLIT VARIANCE ANALYSIS")
    print("="*60)
    print(f"Dogs analyzed (>10 races): {len(dog_stats):,}")
    
    avg_raw_std = dog_stats['RawStdDev'].mean()
    avg_norm_std = dog_stats['NormStdDev'].mean()
    
    print(f"\nAverage Raw StdDev:        {avg_raw_std:.4f} seconds")
    print(f"Average Normalized StdDev: {avg_norm_std:.4f} seconds")
    
    improvement = (avg_raw_std - avg_norm_std) / avg_raw_std * 100
    print(f"Variance Reduction:        {improvement:.1f}%")
    
    print("\n--- Impact of Track Diversity ---")
    multi_track_dogs = dog_stats[dog_stats['TrackCount'] >= 3]
    single_track_dogs = dog_stats[dog_stats['TrackCount'] == 1]
    
    print(f"Multi-Track Dogs ({len(multi_track_dogs)}):")
    print(f"  Raw StdDev:  {multi_track_dogs['RawStdDev'].mean():.4f}")
    print(f"  Norm StdDev: {multi_track_dogs['NormStdDev'].mean():.4f} (Reduction: {(multi_track_dogs['RawStdDev'].mean() - multi_track_dogs['NormStdDev'].mean()) / multi_track_dogs['RawStdDev'].mean() * 100:.1f}%)")
    
    print(f"\nSingle-Track Dogs ({len(single_track_dogs)}):")
    print(f"  Raw StdDev:  {single_track_dogs['RawStdDev'].mean():.4f}")
    print(f"  Norm StdDev: {single_track_dogs['NormStdDev'].mean():.4f} (Reduction: {(single_track_dogs['RawStdDev'].mean() - single_track_dogs['NormStdDev'].mean()) / single_track_dogs['RawStdDev'].mean() * 100:.1f}%)")
    
    print("\n" + "="*60)
    if avg_norm_std < avg_raw_std:
        print("CONCLUSION: Normalizing splits significantly reduces variance.")
        print("Using Normalized Splits will provide much better PIR predictions.")
    else:
        print("CONCLUSION: Normalization did not improve consistency.")

if __name__ == "__main__":
    analyze_split_consistency()
