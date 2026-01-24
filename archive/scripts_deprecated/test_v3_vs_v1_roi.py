"""
Test V3 Pace Model on BSP Strategy
Compare V3 (18 features) vs V1 (8 features) ROI
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
V1_MODEL_PATH = 'models/pace_xgb_model.pkl'
V3_MODEL_PATH = 'models/pace_v3_xgb_model.pkl'

def run_comparison():
    print("Loading Data (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT ge.GreyhoundID, r.RaceID, rm.MeetingDate, t.TrackName, r.Distance, ge.Box,
           ge.FinishTime, ge.Position, ge.StartingPrice, ge.BSP, 
           COALESCE(ge.PrizeMoney, 0) as PrizeMoney
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
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
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
    
    # V3 Additional Features
    df['Roll10'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    df['Std5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std()).fillna(0.5)
    df['Trend'] = df['Roll3'] - df['Roll10']
    
    def exp_weighted_avg(x, span=5):
        return x.shift(1).ewm(span=span, min_periods=3).mean()
    df['ExpRoll5'] = g['NormTime'].transform(lambda x: exp_weighted_avg(x, span=5))
    
    # Track-specific
    df['DogTrack'] = df['GreyhoundID'].astype(str) + '_' + df['TrackName']
    df = df.sort_values(['DogTrack', 'MeetingDate'])
    gt = df.groupby('DogTrack')
    df['TrackRoll3'] = gt['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    df['TrackRoll3'] = df['TrackRoll3'].fillna(df['Roll5'])
    
    # Distance-specific
    df['DistCat'] = pd.cut(df['Distance'], bins=[0, 400, 550, 1000], labels=['Sprint', 'Middle', 'Stay'])
    df['DogDist'] = df['GreyhoundID'].astype(str) + '_' + df['DistCat'].astype(str)
    df = df.sort_values(['DogDist', 'MeetingDate'])
    gd = df.groupby('DogDist')
    df['DistRoll3'] = gd['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    df['DistRoll3'] = df['DistRoll3'].fillna(df['Roll5'])
    
    # Class indicators
    df['CareerPrize'] = df.groupby('GreyhoundID')['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df['LogPrize'] = np.log1p(df['CareerPrize'])
    
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['WinRate'] = df.groupby('GreyhoundID')['IsWin'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.125)
    df['BestTime'] = df.groupby('GreyhoundID')['NormTime'].transform(lambda x: x.shift(1).expanding().min())
    df['BestTime'] = df['BestTime'].fillna(df['Roll5'])
    
    # Filter valid
    df = df.dropna(subset=['Roll5', 'ExpRoll5']).copy()
    
    # Load models
    print("Predicting with V1 and V3 models...")
    with open(V1_MODEL_PATH, 'rb') as f: v1_model = pickle.load(f)
    with open(V3_MODEL_PATH, 'rb') as f: v3_model = pickle.load(f)
    
    # V1 Prediction
    v1_features = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    X_v1 = df[v1_features].copy()
    df['PredPace_V1'] = v1_model.predict(X_v1) + df['TrackDistMedian']
    
    # V3 Prediction
    v3_features = [
        'Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'Roll10',
        'ExpRoll5', 'Std5', 'Trend', 'TrackRoll3', 'DistRoll3',
        'LogPrize', 'WinRate', 'BestTime', 'DaysSince', 'Box', 'Distance'
    ]
    X_v3 = df[v3_features].copy()
    df['PredPace_V3'] = v3_model.predict(X_v3) + df['TrackDistMedian']
    
    # Rankings
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['BSP'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # V1 Ranking
    df = df.sort_values(['RaceKey', 'PredPace_V1'])
    df['Rank_V1'] = df.groupby('RaceKey').cumcount() + 1
    df['NextTime_V1'] = df.groupby('RaceKey')['PredPace_V1'].shift(-1)
    df['Gap_V1'] = df['NextTime_V1'] - df['PredPace_V1']
    
    # V3 Ranking
    df['Rank_V3'] = df.groupby('RaceKey')['PredPace_V3'].rank(method='first', ascending=True)
    min_paces_v3 = df.groupby('RaceKey')['PredPace_V3'].transform('min')
    second_paces_v3 = df.groupby('RaceKey')['PredPace_V3'].transform(lambda x: x.nsmallest(2).iloc[-1] if len(x) >= 2 else x.min())
    df['Gap_V3'] = second_paces_v3 - min_paces_v3
    
    # Test Strategy on Both
    print("\n" + "="*80)
    print("V1 vs V3 COMPARISON: Gap > 0.15s, Prize > 20k, Odds $2-$10 (BSP)")
    print("="*80)
    
    for name, rank_col, gap_col in [("V1 (8 features)", 'Rank_V1', 'Gap_V1'), ("V3 (17 features)", 'Rank_V3', 'Gap_V3')]:
        leaders = df[df[rank_col] == 1].copy()
        
        filt = leaders[
            (leaders[gap_col] >= 0.15) &
            (leaders['CareerPrize'] >= 20000) &
            (leaders['BSP'] >= 2.0) &
            (leaders['BSP'] <= 10.0) &
            leaders['BSP'].notna()
        ].copy()
        
        if len(filt) == 0:
            print(f"\n{name}: No bets")
            continue
            
        filt['Profit'] = filt.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
        
        wins = filt[filt['Position'] == '1'].shape[0]
        date_range = (filt['MeetingDate'].max() - filt['MeetingDate'].min()).days
        bets_per_day = len(filt) / date_range if date_range > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Bets: {len(filt)}")
        print(f"  Bets/Day: {bets_per_day:.1f}")
        print(f"  Strike: {(wins/len(filt))*100:.1f}%")
        print(f"  Profit: {filt['Profit'].sum():.1f}u")
        print(f"  ROI: {(filt['Profit'].sum()/len(filt))*100:.1f}%")

if __name__ == "__main__":
    run_comparison()
