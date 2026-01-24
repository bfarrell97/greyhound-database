"""
Comprehensive Strategy Optimizer for V1 Pace Model
Target: 20% ROI with ~2+ bets/day using BSP
Uses all available database filters
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from itertools import product

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def load_and_prepare():
    print("Loading Data (2020-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
           ge.FinishTime, ge.Position, ge.StartingPrice, ge.BSP, 
           COALESCE(ge.PrizeMoney, 0) as PrizeMoney,
           r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2020-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    df['RaceNumber'] = pd.to_numeric(df['RaceNumber'], errors='coerce').fillna(5)
    
    print("Feature Engineering...")
    
    # Benchmarks
    benchmarks = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    benchmarks.columns = ['TrackName', 'Distance', 'TrackDistMedian']
    df = df.merge(benchmarks, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedian']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # V1 Features
    df['Lag1'] = g['NormTime'].shift(1)
    df['Lag2'] = g['NormTime'].shift(2)
    df['Lag3'] = g['NormTime'].shift(3)
    df['Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter valid
    df = df.dropna(subset=['Roll5']).copy()
    
    # Predict
    print("Predicting...")
    with open(PACE_MODEL_PATH, 'rb') as f: model = pickle.load(f)
    v1_features = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    X = df[v1_features].copy()
    df['PredPace'] = model.predict(X) + df['TrackDistMedian']
    
    # Ranking
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['BSP'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    # Distance category
    df['DistCat'] = pd.cut(df['Distance'], bins=[0, 400, 550, 1000], labels=['Sprint', 'Middle', 'Stay'])
    
    # Calculate date range for bets/day
    date_range = (df['MeetingDate'].max() - df['MeetingDate'].min()).days
    
    return df, date_range

def run_optimization():
    df, date_range = load_and_prepare()
    
    # Filter to leaders only
    leaders = df[df['Rank'] == 1].copy()
    leaders = leaders[leaders['BSP'].notna() & (leaders['BSP'] > 1)]
    
    print(f"\nTotal Pace Leaders with BSP: {len(leaders)}")
    print(f"Date Range: {date_range} days")
    
    # Grid parameters
    gaps = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    prizes = [0, 10000, 15000, 20000, 30000]
    min_odds = [1.50, 1.80, 2.00, 2.20, 2.50, 3.00]
    max_odds = [5, 6, 8, 10, 15]
    dist_cats = ['All', 'Sprint', 'Middle']
    days_since_max = [999, 21, 14, 10, 7]
    field_sizes = [6, 7, 8]
    boxes = ['All', 'Inside', 'Outside']  # Inside: 1-3, Outside: 6-8
    
    results = []
    total_combos = len(gaps) * len(prizes) * len(min_odds) * len(max_odds) * len(dist_cats) * len(days_since_max) * len(field_sizes) * len(boxes)
    
    print(f"Testing {total_combos} combinations...")
    count = 0
    
    for gap in gaps:
        for prize in prizes:
            for min_o in min_odds:
                for max_o in max_odds:
                    for dist in dist_cats:
                        for days in days_since_max:
                            for fs in field_sizes:
                                for box in boxes:
                                    count += 1
                                    if count % 5000 == 0:
                                        print(f"Progress: {count}/{total_combos}...")
                                    
                                    # Apply filters
                                    filt = leaders[
                                        (leaders['Gap'] >= gap) &
                                        (leaders['CareerPrize'] >= prize) &
                                        (leaders['BSP'] >= min_o) &
                                        (leaders['BSP'] <= max_o) &
                                        (leaders['DaysSince'] <= days) &
                                        (leaders['FieldSize'] >= fs)
                                    ]
                                    
                                    # Distance filter
                                    if dist != 'All':
                                        filt = filt[filt['DistCat'] == dist]
                                    
                                    # Box filter
                                    if box == 'Inside':
                                        filt = filt[filt['Box'].isin([1, 2, 3])]
                                    elif box == 'Outside':
                                        filt = filt[filt['Box'].isin([6, 7, 8])]
                                    
                                    if len(filt) < 100:  # Need minimum volume
                                        continue
                                    
                                    # Calculate metrics
                                    filt = filt.copy()
                                    filt['Profit'] = filt.apply(
                                        lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1
                                    )
                                    
                                    wins = filt[filt['Position'] == '1'].shape[0]
                                    bets_per_day = len(filt) / date_range
                                    roi = (filt['Profit'].sum() / len(filt)) * 100
                                    
                                    results.append({
                                        'Gap': gap,
                                        'Prize': prize,
                                        'MinOdds': min_o,
                                        'MaxOdds': max_o,
                                        'Distance': dist,
                                        'DaysSince': days,
                                        'MinField': fs,
                                        'Box': box,
                                        'Bets': len(filt),
                                        'BetsPerDay': bets_per_day,
                                        'Strike': (wins / len(filt)) * 100,
                                        'Profit': filt['Profit'].sum(),
                                        'ROI': roi
                                    })
    
    res_df = pd.DataFrame(results)
    
    # Filter for target: ROI >= 15% AND Bets/Day >= 1.5
    target = res_df[(res_df['ROI'] >= 15) & (res_df['BetsPerDay'] >= 1.5)].sort_values('Profit', ascending=False)
    
    print("\n" + "="*100)
    print("TARGET: ROI >= 15% AND Bets/Day >= 1.5")
    print("="*100)
    
    if len(target) > 0:
        print(f"\nFound {len(target)} viable configurations!")
        print("\nTOP 20 BY PROFIT:")
        print(target.head(20).to_string(index=False))
    else:
        print("No configs meet exact target. Showing best options...")
        
        # Best by ROI with decent volume
        decent = res_df[res_df['BetsPerDay'] >= 1.0].sort_values('ROI', ascending=False)
        print("\nBest ROI with >= 1 bet/day:")
        print(decent.head(15).to_string(index=False))
        
        # Best by Volume with high ROI
        high_roi = res_df[res_df['ROI'] >= 12].sort_values('BetsPerDay', ascending=False)
        print("\nHighest Volume with ROI >= 12%:")
        print(high_roi.head(15).to_string(index=False))
    
    res_df.to_csv('results/strategy_optimization_full.csv', index=False)
    print("\nFull results saved to results/strategy_optimization_full.csv")

if __name__ == "__main__":
    run_optimization()
