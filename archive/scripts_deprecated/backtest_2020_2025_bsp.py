"""
Backtest: Gap > 0.15s, Prize > 20k, Odds $2-$10, BSP
DUAL MODEL: Pace Leader + Split Leader
Period: 2020-2025
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'

def run_backtest():
    print("Loading Data (2020-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
           ge.FinishTime, ge.Position, ge.StartingPrice, ge.BSP, COALESCE(ge.PrizeMoney, 0) as PrizeMoney
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
    
    def parse_price(x):
        try:
            if not x: return np.nan
            return float(str(x).replace('$','').replace('F','').strip())
        except: return np.nan
        
    df['SP'] = df['StartingPrice'].apply(parse_price)
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
    print(f"Total Rows: {len(df)}")
    print(f"BSP Coverage: {df['BSP'].notna().sum()} ({df['BSP'].notna().mean()*100:.1f}%)")
    
    print("Feature Engineering...")
    # PACE benchmark
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # PACE features
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    df = df.dropna(subset=['p_Roll5']).copy()
    
    print("Predicting Pace...")
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    X_pace = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredPace'] = pace_model.predict(X_pace) + df['TrackDistMedianPace']
    
    # Now add SPLIT/PIR prediction
    print("Adding Split Features & Predicting PIR...")
    # Need to reload with Split data
    conn = sqlite3.connect(DB_PATH)
    split_query = """
    SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, ge.Split
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= '2020-01-01' AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    split_df = pd.read_sql_query(split_query, conn)
    conn.close()
    
    split_df['MeetingDate'] = pd.to_datetime(split_df['MeetingDate'])
    split_df['Split'] = pd.to_numeric(split_df['Split'], errors='coerce')
    
    # Merge Split into main df
    df = df.merge(split_df[['GreyhoundID', 'RaceID', 'MeetingDate', 'Split']], 
                  on=['GreyhoundID', 'RaceID', 'MeetingDate'], how='left')
    
    # Split benchmark
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    
    # Split features (need to re-sort and re-group)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # Filter for dogs with split history
    df = df.dropna(subset=['s_Roll5']).copy()
    
    # Predict Split/PIR
    with open(PIR_MODEL_PATH, 'rb') as f: pir_model = pickle.load(f)
    X_split = df[['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredSplit'] = pir_model.predict(X_split) + df['TrackDistMedianSplit']
    
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['SP'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # PACE Ranking
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['PaceRank'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    # SPLIT Ranking (independent)
    df['SplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min', ascending=True)
    
    # Pace Leaders
    pace_leaders = df[df['PaceRank'] == 1].copy()
    
    # Base Filters
    base_mask = (
        (pace_leaders['Gap'] >= 0.15) & 
        (pace_leaders['CareerPrize'] >= 20000) & 
        (pace_leaders['BSP'] >= 2.0) & 
        (pace_leaders['BSP'] <= 10.0) & 
        pace_leaders['BSP'].notna()
    )
    
    # STRATEGY 1: Pace Only
    pace_only = pace_leaders[base_mask].copy()
    pace_only['Profit'] = pace_only.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
    
    # STRATEGY 2: Dual (Pace Leader AND Split Leader)
    dual = pace_leaders[base_mask & (pace_leaders['SplitRank'] == 1)].copy()
    dual['Profit'] = dual.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
    
    print("\n" + "="*80)
    print("DUAL MODEL COMPARISON: Gap > 0.15s, Prize > 20k, Odds $2-$10 (BSP)")
    print("Period: 2020-2025")
    print("="*80)
    
    for name, filt_bsp in [("PACE ONLY", pace_only), ("DUAL (Pace+Split)", dual)]:
        if len(filt_bsp) == 0:
            print(f"\n{name}: No bets")
            continue
            
        date_range = (filt_bsp['MeetingDate'].max() - filt_bsp['MeetingDate'].min()).days
        bets_per_day = len(filt_bsp) / date_range if date_range > 0 else 0
        filt_bsp['Year'] = filt_bsp['MeetingDate'].dt.year
        
        wins = filt_bsp[filt_bsp['Position'] == '1'].shape[0]
        print(f"\n{name}:")
        print(f"  Bets: {len(filt_bsp)}")
        print(f"  Bets/Day: {bets_per_day:.1f}")
        print(f"  Strike: {(wins/len(filt_bsp))*100:.1f}%")
        print(f"  Profit: {filt_bsp['Profit'].sum():.1f}u")
        print(f"  ROI: {(filt_bsp['Profit'].sum()/len(filt_bsp))*100:.1f}%")
        print(f"  Avg BSP: ${filt_bsp['BSP'].mean():.2f}")
        
        print(f"\n  YEARLY BREAKDOWN:")
        print(f"  {'Year':<6} | {'Bets':<6} | {'Strike %':<9} | {'Profit':<10} | {'ROI %':<8}")
        print("  " + "-"*50)
        
        for year in sorted(filt_bsp['Year'].unique()):
            yr = filt_bsp[filt_bsp['Year'] == year]
            yr_wins = yr[yr['Position'] == '1'].shape[0]
            yr_profit = yr['Profit'].sum()
            yr_roi = (yr_profit / len(yr)) * 100 if len(yr) > 0 else 0
            print(f"  {year:<6} | {len(yr):<6} | {(yr_wins/len(yr))*100:<9.1f} | {yr_profit:<10.1f} | {yr_roi:<8.1f}")

if __name__ == "__main__":
    run_backtest()
