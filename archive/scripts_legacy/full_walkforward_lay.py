"""
Full Rolling Walk-Forward Backtest (2023-2025)
Simulates monthly retraining and trading to verify consistency.
Strategy: Lay "Odds-On" False Favorites (Mg > 0.1, Odds < 2.25)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlite3
from datetime import timedelta

# Config
DB_PATH = 'greyhound_racing.db'
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
TRAIN_MONTHS = 24
RETRAIN_MONTHS = 1 # Step size
STAKE = 2
COMM = 0.05

def load_data():
    print("Loading Data (2021-2025)...")
    with open('tier1_tracks.txt', 'r') as f:
        safe_tracks = [line.strip() for line in f if line.strip()]
    
    conn = sqlite3.connect(DB_PATH)
    placeholders = ',' .join('?' for _ in safe_tracks)
    query = f"""
    SELECT
        ge.GreyhoundID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND t.TrackName IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=safe_tracks)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # Feature Eng
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    return df.dropna(subset=['DogNormTimeAvg', 'Odds'])

def train_and_predict(train_df, test_df):
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, tree_method='hist')
    model.fit(train_df[features], train_df['NormTime'])
    
    test_df = test_df.copy()
    test_df['PredOverall'] = model.predict(test_df[features])
    return test_df

def apply_strategy(test_df):
    # Ranking
    test_df['PredRank'] = test_df.groupby('RaceID')['PredOverall'].rank(method='min')
    
    rank1s = test_df[test_df['PredRank'] == 1].copy()
    rank2s = test_df[test_df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    # Strategy Filter: Odds < 2.25, Margin > 0.1
    trades = candidates[
        (candidates['Margin'] > 0.1) & 
        (candidates['Odds'] < 2.25)
    ].copy()
    
    return trades

def run_rolling_backtest(df):
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    total_trades = []
    
    print("\nStarting Rolling Walk-Forward...")
    print(f"{'Month':<10} {'Train Size':<10} {'Test Size':<10} {'Bets':<5} {'Profit':<10} {'ROI':<6}")
    print("-" * 60)
    
    for date in dates:
        month_start = date
        month_end = date + pd.offsets.MonthEnd(0)
        
        train_start = month_start - pd.DateOffset(months=TRAIN_MONTHS)
        
        # Slicing
        train_data = df[
            (df['MeetingDate'] >= train_start) & 
            (df['MeetingDate'] < month_start)
        ]
        
        # Test Data is THIS month
        test_data = df[
            (df['MeetingDate'] >= month_start) & 
            (df['MeetingDate'] <= month_end)
        ]
        
        if len(test_data) == 0: continue
        
        # Train & Predict
        test_w_preds = train_and_predict(train_data, test_data)
        
        # Trade
        trades = apply_strategy(test_w_preds)
        
        # Calc P&L
        if len(trades) > 0:
            trades['PnL'] = np.where( trades['IsWin'] == 0, STAKE * (1-COMM), -1 * STAKE * (trades['Odds'] - 1) )
            trades['Liability'] = (trades['Odds'] - 1) * STAKE
            
            month_profit = trades['PnL'].sum()
            month_risk = trades['Liability'].sum()
            month_roi = month_profit / month_risk * 100 if month_risk > 0 else 0
            
            total_trades.append(trades)
            
            print(f"{date.strftime('%Y-%m'):<10} {len(train_data):<10} {len(test_data):<10} {len(trades):<5} ${month_profit:<9.0f} {month_roi:<6.1f}")
        else:
            print(f"{date.strftime('%Y-%m'):<10} {len(train_data):<10} {len(test_data):<10} 0     $0         0.0")

    if len(total_trades) > 0:
        all_trades = pd.concat(total_trades)
        
        print("\n[ FINAL RESULTS ]")
        gross = all_trades['PnL'].sum()
        risk = all_trades['Liability'].sum()
        roi = gross / risk * 100
        
        print(f"Total Bets: {len(all_trades)}")
        print(f"Total Profit: ${gross:.2f}")
        print(f"Total ROI: {roi:.2f}%")
        
        # Monthly Stats
        all_trades['Month'] = all_trades['MeetingDate'].dt.to_period('M')
        monthly = all_trades.groupby('Month')['PnL'].sum()
        positive_months = (monthly > 0).sum()
        print(f"Winning Months: {positive_months} / {len(monthly)} ({positive_months/len(monthly)*100:.1f}%)")

if __name__ == "__main__":
    df = load_data()
    run_rolling_backtest(df)
