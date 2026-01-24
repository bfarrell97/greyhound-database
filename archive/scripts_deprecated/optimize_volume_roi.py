"""
Optimize for Target: ~15% ROI + ~10 bets/day at BSP
Grid search over Gap, Prize, and Odds thresholds to find the sweet spot.
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def load_and_prep():
    print("Loading Data (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
           ge.FinishTime, ge.Position, ge.StartingPrice, ge.BSP, COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            return float(str(x).replace('$','').replace('F','').strip())
        except: return np.nan
        
    df['SP'] = df['StartingPrice'].apply(parse_price)
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
    print("Feature Engineering...")
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    df = df.dropna(subset=['p_Roll5']).copy()
    
    print("Predicting...")
    with open(PACE_MODEL_PATH, 'rb') as f: model = pickle.load(f)
    X = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredPace'] = model.predict(X) + df['TrackDistMedianPace']
    
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['SP'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    # Calculate days in dataset for bets/day calculation
    date_range = (df['MeetingDate'].max() - df['MeetingDate'].min()).days
    
    return df, date_range

def optimize(df, date_range):
    leaders = df[df['Rank'] == 1].copy()
    
    # Grid
    gaps = [0.05, 0.08, 0.10, 0.12, 0.15]
    prizes = [0, 5000, 10000, 15000, 20000]
    min_odds = [1.50, 1.80, 2.00, 2.20]
    max_odds = [15, 20, 30, 50]
    
    results = []
    
    print("Running Grid Search...")
    for gap in gaps:
        for prize in prizes:
            for min_o in min_odds:
                for max_o in max_odds:
                    # BSP filter
                    filt = leaders[
                        (leaders['Gap'] >= gap) & 
                        (leaders['CareerPrize'] >= prize) & 
                        (leaders['BSP'] >= min_o) & 
                        (leaders['BSP'] <= max_o) & 
                        leaders['BSP'].notna()
                    ]
                    
                    if len(filt) < 50: continue
                    
                    filt = filt.copy()
                    filt['Profit'] = filt.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
                    
                    wins = filt[filt['Position'] == '1'].shape[0]
                    roi = (filt['Profit'].sum() / len(filt)) * 100
                    bets_per_day = len(filt) / date_range
                    
                    results.append({
                        'Gap': gap,
                        'Prize': prize,
                        'MinOdds': min_o,
                        'MaxOdds': max_o,
                        'Bets': len(filt),
                        'BetsPerDay': bets_per_day,
                        'Strike': (wins / len(filt)) * 100,
                        'Profit': filt['Profit'].sum(),
                        'ROI': roi
                    })
    
    res_df = pd.DataFrame(results)
    
    # Filter for target: ROI >= 10% AND Bets/Day >= 5
    viable = res_df[(res_df['ROI'] >= 10) & (res_df['BetsPerDay'] >= 5)].sort_values('Profit', ascending=False)
    
    print("\n" + "="*100)
    print("TARGET: ~15% ROI + ~10 Bets/Day (BSP)")
    print("="*100)
    
    if len(viable) > 0:
        print("\nVIABLE CONFIGURATIONS (ROI >= 10%, Bets/Day >= 5):")
        print("-"*100)
        print(viable.head(20).to_string(index=False))
    else:
        print("\nNo configs meet target. Showing best options:")
        # Best by ROI with decent volume
        decent_vol = res_df[res_df['BetsPerDay'] >= 3].sort_values('ROI', ascending=False)
        print("\nBest ROI with >= 3 bets/day:")
        print(decent_vol.head(10).to_string(index=False))
        
        # Best by Volume with positive ROI
        pos_roi = res_df[res_df['ROI'] > 0].sort_values('BetsPerDay', ascending=False)
        print("\nHighest Volume with positive ROI:")
        print(pos_roi.head(10).to_string(index=False))
    
    res_df.to_csv('results/volume_roi_optimization.csv', index=False)
    print("\nFull results saved to results/volume_roi_optimization.csv")

if __name__ == "__main__":
    df, date_range = load_and_prep()
    print(f"Date Range: {date_range} days")
    optimize(df, date_range)
