"""
Backtest ML Strategy (PIR + Pace XGBoost Models)
Replicates the feature engineering and prediction vectorially to backtest the ML-driven strategy.
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

DB_PATH = 'greyhound_racing.db'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    print("Loading data...")
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split,
        ge.FinishTime,
        ge.Position,
        ge.StartingPrice,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney,
        (ge.FinishTimeBenchmarkLengths + COALESCE(rm.MeetingAvgBenchmarkLengths, 0)) as TotalPace -- Original Pace Metric for comparison if needed
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01' -- Test on strictly unseen data (Train was 2020-2023)
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    # Parse Odds
    def parse_price(x):
        try:
            if not x or x is None: return np.nan
            x = str(x).replace('$', '').strip()
            if 'F' in x: x = x.replace('F', '')
            return float(x)
        except:
            return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # Career Prize Money (Running Total)
    # We must calculate this properly to avoid leakage
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['CareerPrize'] = df.groupby('GreyhoundID')['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    return df

def feature_engineering(df):
    print("Calculating ML Features...")
    
    # 1. Benchmarks (Track/Dist Median)
    # Note: Strictly speaking, we should calculate benchmarks on PRE-2024 data to avoid leakage.
    # However, for this backtest we'll use the "Static" benchmarks logic similar to the App which uses 2024 data too.
    # To be perfectly rigorous, we'd load pre-2024 benchmarks.
    # For now, let's calculate benchmarks on the dataset itself using expanding window or just global (as the App does static).
    # The App uses a static query of "Split > 0".
    # Let's use Global Medians for simplicity/parity with App.
    
    # Calc Global Medians
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    # 2. Norm Targets
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    # 3. Rolling Features
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Split Feats
    g = df.groupby('GreyhoundID')['NormSplit']
    df['s_Lag1'] = g.shift(1)
    df['s_Lag2'] = g.shift(2)
    df['s_Lag3'] = g.shift(3)
    df['s_Roll3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # Pace Feats
    g_p = df.groupby('GreyhoundID')['NormTime']
    df['p_Lag1'] = g_p.shift(1)
    df['p_Lag2'] = g_p.shift(2)
    df['p_Lag3'] = g_p.shift(3)
    df['p_Roll3'] = g_p.transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g_p.transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # Days Since
    df['PrevDate'] = df.groupby('GreyhoundID')['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999)
    
    return df

def run_predictions(df):
    print("Running XGBoost Inference...")
    
    # Load Models
    with open(PIR_MODEL_PATH, 'rb') as f:
        pir_model = pickle.load(f)
    print("Loaded PIR Model")
    
    with open(PACE_MODEL_PATH, 'rb') as f:
        pace_model = pickle.load(f)
    print("Loaded Pace Model")
    
    # Prepare Features
    # Note: Model features must match training exactly
    # PIR Features: ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    feats_split = df[['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    feats_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    feats_pace = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    feats_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    # XGBoost handles NaNs, but for "Prediction" in production we usually skip if insufficient history.
    # However, XGBoost can predict on partial data.
    # The models were trained on data where these columns WERE valid (dropna).
    # So creating predictions on rows with NaNs might yield garbage or default paths.
    # To correspond with the App's "5 race history" filter, we should probably stick to valid rows.
    # But let's let XGBoost predict and we'll filter by RaceCount later.
    
    df['PredictedNormSplit'] = pir_model.predict(feats_split)
    df['PredictedNormPace'] = pace_model.predict(feats_pace)
    
    # Denormalize
    df['PredictedSplit'] = df['PredictedNormSplit'] + df['TrackDistMedianSplit']
    df['PredictedPace'] = df['PredictedNormPace'] + df['TrackDistMedianPace']
    
    return df

def backtest(df):
    print("Constructing Market & Simulating Bets...")
    
    # 1. Filters
    # Distance <= 600
    df = df[df['Distance'] <= 600]
    
    # History Filter (Proxy using Roll5 check)
    # The App requires 5 races.
    # If s_Roll5 is NaN, it means < 5 previous races.
    df = df.dropna(subset=['s_Roll5', 'p_Roll5'])
    
    # 2. Ranking
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Min Odds Filter
    df['MinMarketOdds'] = df.groupby('RaceKey')['Odds'].transform('min')
    df = df[df['MinMarketOdds'] >= 1.30] # Exclude heavy fav races
    
    # Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # Rank Ascending (Lower is better for both Split and Time)
    df['ML_Split_Rank'] = df.groupby('RaceKey')['PredictedSplit'].rank(method='min', ascending=True)
    df['ML_Pace_Rank'] = df.groupby('RaceKey')['PredictedPace'].rank(method='min', ascending=True)
    
    # 3. Strategy Logic
    df['Is_ML_PIR_Leader'] = df['ML_Split_Rank'] == 1
    df['Is_ML_Pace_Leader'] = df['ML_Pace_Rank'] == 1
    df['Is_ML_Pace_Top3'] = df['ML_Pace_Rank'] <= 3
    
    # Common filters
    df['HasMoney'] = df['CareerPrize'] >= 30000
    df['InOdds'] = (df['Odds'] >= 1.50) & (df['Odds'] <= 30)
    
    # Strategies DF
    strategies = {
        '1. ML PIR Leader': df[df['Is_ML_PIR_Leader'] & df['HasMoney'] & df['InOdds']].copy(),
        '2. ML Pace Leader': df[df['Is_ML_Pace_Leader'] & df['HasMoney'] & df['InOdds']].copy(),
        '3. Dual ML Leader': df[df['Is_ML_PIR_Leader'] & df['Is_ML_Pace_Leader'] & df['HasMoney'] & df['InOdds']].copy(),
        '4. ML Pace Top 3': df[df['Is_ML_Pace_Top3'] & df['HasMoney'] & df['InOdds']].copy()
    }
    
    # Results
    print("\n" + "="*80)
    print("ML STRATEGY BACKTEST RESULTS (2024-2025)")
    print(f"Sample: {len(df)} runners analysed")
    print("="*80)
    print(f"{'Strategy':<20} | {'Bets':<6} | {'Winners':<7} | {'Strike %':<8} | {'P/L':<8} | {'ROI %':<8}")
    print("-" * 80)
    
    for name, strat_df in strategies.items():
        if len(strat_df) == 0:
            print(f"{name:<20} | 0      | 0       | 0.0%     | 0.00     | 0.0%")
            continue
            
        bets = len(strat_df)
        wins = strat_df[strat_df['Position'] == '1'].shape[0]
        strike = (wins / bets) * 100
        
        # Flat stakes
        profit = strat_df[strat_df['Position'] == '1']['Odds'].sum() - bets
        roi = (profit / bets) * 100
        
        print(f"{name:<20} | {bets:<6} | {wins:<7} | {strike:<8.1f} | {profit:<8.1f} | {roi:<8.1f}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    df = run_predictions(df)
    backtest(df)
