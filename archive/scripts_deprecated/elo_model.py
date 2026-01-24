"""
Elo Rating Model for Greyhound Racing
======================================
Implements an Elo-style rating system where:
- Every dog starts with rating 1500
- Winners gain rating, losers lose rating
- Amount gained/lost depends on expected vs actual outcome
- K-factor determines sensitivity to recent results

Backtest: 2020-2024 train, 2025 test
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Elo parameters
INITIAL_RATING = 1500
K_FACTOR = 32  # How much ratings change per race
DECAY_DAYS = 60  # Ratings decay toward baseline after inactivity

def calculate_elo():
    print("="*70)
    print("ELO RATING MODEL FOR GREYHOUND RACING")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load all race data
    print("\n[1/4] Loading race data...")
    query = """
    SELECT ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP,
           rm.MeetingDate, t.TrackName, r.Distance, r.RaceNumber,
           ge.PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL
      AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, r.RaceNumber
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = df['Position'] == 1
    
    print(f"Loaded {len(df):,} race entries")
    print(f"Date range: {df['MeetingDate'].min()} to {df['MeetingDate'].max()}")
    
    # [2/4] Calculate Elo ratings
    print("\n[2/4] Calculating Elo ratings...")
    
    # Track ratings and last race date for each dog
    ratings = defaultdict(lambda: INITIAL_RATING)
    last_race = {}
    
    # Store predictions for each race
    predictions = []
    
    races_processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 2:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        
        # Get current ratings for each dog (with decay for inactive dogs)
        race_ratings = {}
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            current_rating = ratings[dog_id]
            
            # Apply decay for inactivity
            if dog_id in last_race:
                days_since = (race_date - last_race[dog_id]).days
                if days_since > DECAY_DAYS:
                    decay_factor = min(0.9, 1 - (days_since - DECAY_DAYS) / 365)
                    current_rating = INITIAL_RATING + (current_rating - INITIAL_RATING) * decay_factor
            
            race_ratings[dog_id] = current_rating
        
        # Calculate expected win probabilities (softmax of ratings)
        total_exp = sum(np.exp(r / 400) for r in race_ratings.values())
        expected = {dog: np.exp(race_ratings[dog] / 400) / total_exp for dog in race_ratings}
        
        # Find predicted winner (highest rating)
        predicted_winner = max(race_ratings, key=race_ratings.get)
        
        # Store prediction info for this race (only for 2025 test set)
        if race_date >= datetime(2025, 1, 1):
            for _, row in race_df.iterrows():
                dog_id = row['GreyhoundID']
                is_leader = (dog_id == predicted_winner)
                predictions.append({
                    'RaceID': race_id,
                    'GreyhoundID': dog_id,
                    'Date': race_date,
                    'Rating': race_ratings[dog_id],
                    'ExpectedWin': expected[dog_id],
                    'IsLeader': is_leader,
                    'Won': row['Won'],
                    'BSP': row['BSP'],
                    'Distance': row['Distance'],
                    'PrizeMoney': row['PrizeMoney']
                })
        
        # Update ratings based on actual results
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            actual = 1.0 if row['Won'] else 0.0
            exp = expected[dog_id]
            
            # Elo update: R' = R + K * (actual - expected)
            ratings[dog_id] = race_ratings[dog_id] + K_FACTOR * (actual - exp)
            last_race[dog_id] = race_date
        
        races_processed += 1
        if races_processed % 10000 == 0:
            print(f"  Processed {races_processed:,} races...")
    
    print(f"  Total: {races_processed:,} races processed")
    print(f"  Unique dogs rated: {len(ratings):,}")
    
    # [3/4] Analyze 2025 predictions
    print("\n[3/4] Analyzing 2025 predictions...")
    
    pred_df = pd.DataFrame(predictions)
    pred_df['BSP'] = pd.to_numeric(pred_df['BSP'], errors='coerce')
    
    # Get race leaders (highest Elo per race)
    leaders = pred_df[pred_df['IsLeader']].copy()
    print(f"Total race leaders in 2025: {len(leaders):,}")
    
    # Calculate gap in ratings
    for race_id, race_group in pred_df.groupby('RaceID'):
        sorted_ratings = race_group['Rating'].sort_values(ascending=False)
        if len(sorted_ratings) >= 2:
            gap = sorted_ratings.iloc[0] - sorted_ratings.iloc[1]
            pred_df.loc[pred_df['RaceID'] == race_id, 'RatingGap'] = gap
    
    leaders = pred_df[pred_df['IsLeader']].copy()
    
    # [4/4] Backtest results
    print("\n[4/4] Backtest Results...")
    print("="*70)
    
    # Overall Elo leader performance
    print("\nELO LEADER PERFORMANCE (All 2025 races):")
    total_races = len(leaders)
    wins = leaders['Won'].sum()
    sr = wins / total_races * 100
    print(f"  Races: {total_races:,}")
    print(f"  Wins: {wins:,} ({sr:.1f}%)")
    
    # Filter by BSP range $3-$8
    filtered = leaders[(leaders['BSP'] >= 3) & (leaders['BSP'] <= 8)].copy()
    print(f"\nELO LEADER @ $3-$8 BSP:")
    print(f"  Bets: {len(filtered):,}")
    wins = filtered['Won'].sum()
    sr = wins / len(filtered) * 100 if len(filtered) > 0 else 0
    returns = filtered[filtered['Won']]['BSP'].sum()
    profit = returns - len(filtered)
    roi = (profit / len(filtered)) * 100 if len(filtered) > 0 else 0
    print(f"  Wins: {wins:,} ({sr:.1f}%)")
    print(f"  Profit: {profit:.2f}u")
    print(f"  ROI: {roi:.1f}%")
    
    # Filter by Rating Gap >= 50 (equivalent to ~0.1s pace gap)
    gap_filtered = filtered[filtered['RatingGap'] >= 50].copy()
    print(f"\nELO LEADER @ $3-$8 + RATING GAP >= 50:")
    if len(gap_filtered) > 0:
        print(f"  Bets: {len(gap_filtered):,}")
        wins = gap_filtered['Won'].sum()
        sr = wins / len(gap_filtered) * 100
        returns = gap_filtered[gap_filtered['Won']]['BSP'].sum()
        profit = returns - len(gap_filtered)
        roi = (profit / len(gap_filtered)) * 100
        print(f"  Wins: {wins:,} ({sr:.1f}%)")
        print(f"  Profit: {profit:.2f}u")
        print(f"  ROI: {roi:.1f}%")
    
    # Try different gap thresholds
    print("\n" + "="*70)
    print("ROI BY RATING GAP THRESHOLD")
    print("="*70)
    print(f"{'Gap Threshold':<15} {'Bets':>8} {'Strike%':>10} {'Profit':>10} {'ROI':>8}")
    print("-"*60)
    
    for gap in [0, 25, 50, 75, 100, 150, 200]:
        subset = filtered[filtered['RatingGap'] >= gap]
        if len(subset) < 50:
            continue
        wins = subset['Won'].sum()
        sr = wins / len(subset) * 100
        returns = subset[subset['Won']]['BSP'].sum()
        profit = returns - len(subset)
        roi = (profit / len(subset)) * 100
        print(f"{gap:>10} pts {len(subset):>8,} {sr:>9.1f}% {profit:>9.1f}u {roi:>7.1f}%")
    
    # Compare to middle distance filter
    print("\n" + "="*70)
    print("ELO + MIDDLE DISTANCE (400-550m) + Gap >= 50")
    print("="*70)
    
    mid_dist = gap_filtered[(gap_filtered['Distance'] >= 400) & (gap_filtered['Distance'] < 550)]
    if len(mid_dist) > 0:
        wins = mid_dist['Won'].sum()
        sr = wins / len(mid_dist) * 100
        returns = mid_dist[mid_dist['Won']]['BSP'].sum()
        profit = returns - len(mid_dist)
        roi = (profit / len(mid_dist)) * 100
        print(f"  Bets: {len(mid_dist):,}")
        print(f"  Wins: {wins:,} ({sr:.1f}%)")
        print(f"  Profit: {profit:.2f}u")
        print(f"  ROI: {roi:.1f}%")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    return pred_df

if __name__ == "__main__":
    calculate_elo()
