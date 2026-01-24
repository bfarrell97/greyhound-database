"""
Backtest Pace Back Strategy V1
==============================
Strategy: Gap >= 0.15, Distance 400-550m, Odds $3-$8
Period: March 2025 - November 2025
Prices: SP, BSP, Price5Min
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

def backtest():
    print("="*70)
    print("BACKTEST: Pace Back Strategy V1 (Mar-Nov 2025)")
    print("="*70)
    print("Filters: Gap >= 0.15, Distance 400-550m, Odds $3-$8")
    print()
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Load pace model
    with open('models/pace_xgb_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    pace_model = artifacts['model']
    
    # Load benchmarks
    bench_df = pd.read_sql_query(
        "SELECT TrackName, Distance, MedianTime AS TrackDistMedian FROM Benchmarks",
        conn
    )
    
    # Load race data for Mar-Nov 2025
    print("Loading Mar-Nov 2025 data...")
    query = """
    SELECT ge.EntryID, ge.RaceID, g.GreyhoundID, g.GreyhoundName, 
           r.RaceNumber, r.Distance, t.TrackName, rm.MeetingDate,
           ge.Box, ge.Position, ge.StartingPrice, ge.BSP, ge.Price5Min
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate BETWEEN '2025-03-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(df):,} entries")
    
    # Get historical data for features (pre-March 2025)
    print("Loading historical data for features...")
    hist_query = """
    SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.GreyhoundID IN ({})
      AND rm.MeetingDate < '2025-12-01'
      AND ge.FinishTime IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate
    """.format(','.join(str(x) for x in df['GreyhoundID'].unique()))
    
    hist_df = pd.read_sql_query(hist_query, conn)
    conn.close()
    
    # Calculate NormTime for history
    hist_df = hist_df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
    hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['TrackDistMedian']
    hist_df = hist_df.dropna(subset=['NormTime'])
    
    # Calculate rolling features per dog
    print("Calculating features...")
    hist_df = hist_df.sort_values(['GreyhoundID', 'MeetingDate'])
    hist_df['MeetingDate'] = pd.to_datetime(hist_df['MeetingDate'])
    g = hist_df.groupby('GreyhoundID')
    hist_df['Lag1'] = g['NormTime'].shift(0)
    hist_df['Lag2'] = g['NormTime'].shift(1)
    hist_df['Lag3'] = g['NormTime'].shift(2)
    hist_df['Roll3'] = g['NormTime'].transform(lambda x: x.rolling(3, min_periods=3).mean())
    hist_df['Roll5'] = g['NormTime'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    hist_df['PrevDate'] = g['MeetingDate'].shift(0)
    
    # For each race in test period, get features as of that date
    results = []
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    # Group by race and process
    print("Processing races...")
    race_groups = df.groupby('RaceID')
    processed = 0
    
    for race_id, race_df in race_groups:
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        
        # Skip if not middle distance
        if distance < 400 or distance >= 550:
            continue
        
        # Get features for each dog in this race (as of race date)
        race_features = []
        for _, row in race_df.iterrows():
            dog_id = row['GreyhoundID']
            
            # Get dog's history before this race
            dog_hist = hist_df[(hist_df['GreyhoundID'] == dog_id) & 
                               (hist_df['MeetingDate'] < race_date)]
            
            if len(dog_hist) < 5:
                continue
            
            latest = dog_hist.iloc[-1]
            days_since = (race_date - latest['PrevDate']).days
            if pd.isna(days_since):
                days_since = 30
            days_since = min(days_since, 60)
            
            race_features.append({
                'EntryID': row['EntryID'],
                'GreyhoundID': dog_id,
                'GreyhoundName': row['GreyhoundName'],
                'Box': row['Box'],
                'Distance': distance,
                'Position': row['Position'],
                'SP': row['StartingPrice'],
                'BSP': row['BSP'],
                'P5': row['Price5Min'],
                'Lag1': latest['Lag1'],
                'Lag2': latest['Lag2'],
                'Lag3': latest['Lag3'],
                'Roll3': latest['Roll3'],
                'Roll5': latest['Roll5'],
                'DaysSince': days_since
            })
        
        if len(race_features) < 2:
            continue
        
        race_feat_df = pd.DataFrame(race_features)
        
        # Predict pace
        X = race_feat_df[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']]
        race_feat_df['PredPace'] = pace_model.predict(X)
        
        # Rank and calculate gap
        race_feat_df = race_feat_df.sort_values('PredPace')
        race_feat_df['Rank'] = range(1, len(race_feat_df) + 1)
        
        if len(race_feat_df) >= 2:
            gap = race_feat_df['PredPace'].iloc[1] - race_feat_df['PredPace'].iloc[0]
        else:
            gap = 0
        
        # Get pace leader
        leader = race_feat_df.iloc[0].copy()
        leader['Gap'] = gap
        results.append(leader)
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} races...")
    
    print(f"\nTotal races processed: {processed}")
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    print(f"Pace leaders found: {len(results_df)}")
    
    # Apply strategy filters
    def safe_float(x):
        try:
            if pd.isna(x) or x in [0, '0', None, 'None']:
                return None
            return float(x)
        except:
            return None
    
    results_df['SP'] = results_df['SP'].apply(safe_float)
    results_df['BSP'] = results_df['BSP'].apply(safe_float)
    results_df['P5'] = results_df['P5'].apply(safe_float)
    
    # Check if won
    results_df['Won'] = results_df['Position'].apply(lambda x: str(x).strip() == '1')
    
    # Filter: Gap >= 0.15
    filtered = results_df[results_df['Gap'] >= 0.15].copy()
    print(f"After Gap >= 0.15: {len(filtered)}")
    
    # Test at different price points
    print("\n" + "="*70)
    print("RESULTS BY PRICE TYPE")
    print("="*70)
    
    def calc_roi(df, price_col, label):
        valid = df[(df[price_col].notna()) & 
                   (df[price_col] >= 3.0) & 
                   (df[price_col] <= 8.0)].copy()
        
        if len(valid) == 0:
            print(f"\n{label}: No valid bets")
            return
        
        stakes = len(valid) * 1.0  # $1 per bet
        wins = valid['Won'].sum()
        returns = valid[valid['Won']][price_col].sum()
        profit = returns - stakes
        roi = (profit / stakes) * 100 if stakes > 0 else 0
        avg_odds = valid[price_col].mean()
        strike_rate = (wins / len(valid)) * 100 if len(valid) > 0 else 0
        
        print(f"\n{label}:")
        print(f"  Bets: {len(valid)}")
        print(f"  Wins: {wins} ({strike_rate:.1f}%)")
        print(f"  Avg Odds: ${avg_odds:.2f}")
        print(f"  Profit: ${profit:.2f}")
        print(f"  ROI: {roi:.1f}%")
        
        return {'bets': len(valid), 'wins': wins, 'profit': profit, 'roi': roi}
    
    sp_results = calc_roi(filtered, 'SP', 'Starting Price (SP)')
    bsp_results = calc_roi(filtered, 'BSP', 'Betfair Starting Price (BSP)')
    p5_results = calc_roi(filtered, 'P5', 'Price 5 Minutes Before (P5)')
    
    # Monthly breakdown for BSP
    print("\n" + "="*70)
    print("BSP MONTHLY BREAKDOWN")
    print("="*70)
    
    filtered['Month'] = pd.to_datetime(results_df.loc[filtered.index, 'MeetingDate'] if 'MeetingDate' in results_df.columns else '2025-01-01').dt.strftime('%Y-%m')
    
    bsp_valid = filtered[(filtered['BSP'].notna()) & 
                         (filtered['BSP'] >= 3.0) & 
                         (filtered['BSP'] <= 8.0)]
    
    for month in sorted(bsp_valid['Month'].unique()):
        month_data = bsp_valid[bsp_valid['Month'] == month]
        stakes = len(month_data)
        wins = month_data['Won'].sum()
        returns = month_data[month_data['Won']]['BSP'].sum()
        profit = returns - stakes
        roi = (profit / stakes) * 100 if stakes > 0 else 0
        print(f"{month}: {stakes} bets, {wins} wins, ${profit:.2f} ({roi:+.1f}%)")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    backtest()
