"""
Backtest ML Strategy using BSP (Betfair Starting Price)
Compares performance using BSP vs the standard SP (Starting Price).
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'

def run_bsp_backtest():
    print("Loading Data (2024-2025)...")
    conn = sqlite3.connect(DB_PATH)
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
        ge.BSP,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    
    df['SP'] = df['StartingPrice'].apply(parse_price)
    df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
    
    print(f"BSP Coverage: {df['BSP'].notna().sum()} / {len(df)} ({df['BSP'].notna().mean()*100:.1f}%)")
    
    # Feature Engineering
    print("Feature Engineering...")
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # SPLIT features
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # PACE features
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter valid
    df = df.dropna(subset=['p_Roll5', 's_Roll5']).copy()
    
    # Predict
    print("Predicting...")
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    with open(PIR_MODEL_PATH, 'rb') as f: pir_model = pickle.load(f)
    
    # Pace
    X_pace = df[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormPace'] = pace_model.predict(X_pace)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    # Split
    X_split = df[['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
    X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormSplit'] = pir_model.predict(X_split)
    df['PredSplit'] = df['PredNormSplit'] + df['TrackDistMedianSplit']
    
    # Rank
    print("Ranking...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    df['FieldSize'] = df.groupby('RaceKey')['SP'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    df['PaceRank'] = df.groupby('RaceKey')['PredPace'].rank(method='min', ascending=True)
    df['PIRRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min', ascending=True)
    
    # Calculate Gaps
    df = df.sort_values(['RaceKey', 'PredPace'])
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    # Base filters
    df = df[df['Distance'] <= 600]
    
    # Strategy: Pace Leader
    leaders = df[df['PaceRank'] == 1].copy()
    
    print("\n" + "="*80)
    print("ML PACE LEADER STRATEGY - BSP vs SP Comparison")
    print("="*80)
    
    # Filter for valid BSP
    leaders_bsp = leaders[leaders['BSP'].notna() & (leaders['BSP'] > 1)].copy()
    leaders_sp = leaders[leaders['SP'].notna() & (leaders['SP'] > 1)].copy()
    
    print(f"\nBets with valid BSP: {len(leaders_bsp)}")
    print(f"Bets with valid SP:  {len(leaders_sp)}")
    
    # Calc Profit
    leaders_bsp['ProfitBSP'] = leaders_bsp.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
    leaders_sp['ProfitSP'] = leaders_sp.apply(lambda x: (x['SP'] - 1) if x['Position'] == '1' else -1, axis=1)
    
    # BSP Results
    bsp_wins = leaders_bsp[leaders_bsp['Position'] == '1'].shape[0]
    bsp_strike = (bsp_wins / len(leaders_bsp)) * 100 if len(leaders_bsp) > 0 else 0
    bsp_profit = leaders_bsp['ProfitBSP'].sum()
    bsp_roi = (bsp_profit / len(leaders_bsp)) * 100 if len(leaders_bsp) > 0 else 0
    
    # SP Results
    sp_wins = leaders_sp[leaders_sp['Position'] == '1'].shape[0]
    sp_strike = (sp_wins / len(leaders_sp)) * 100 if len(leaders_sp) > 0 else 0
    sp_profit = leaders_sp['ProfitSP'].sum()
    sp_roi = (sp_profit / len(leaders_sp)) * 100 if len(leaders_sp) > 0 else 0
    
    print("\n" + "-"*60)
    print(f"{'Strategy':<20} | {'Bets':<6} | {'Strike %':<9} | {'Profit':<10} | {'ROI %':<8}")
    print("-"*60)
    print(f"{'Pace Leader (BSP)':<20} | {len(leaders_bsp):<6} | {bsp_strike:<9.1f} | {bsp_profit:<10.1f} | {bsp_roi:<8.1f}")
    print(f"{'Pace Leader (SP)':<20} | {len(leaders_sp):<6} | {sp_strike:<9.1f} | {sp_profit:<10.1f} | {sp_roi:<8.1f}")
    
    # Short Course Dominant (Gap > 0.15, Dist < 400)
    print("\n" + "="*80)
    print("SHORT COURSE DOMINANT STRATEGY (Gap > 0.15s, Dist < 400m) - BSP vs SP")
    print("="*80)
    
    short_dom = leaders[
        (leaders['Distance'] < 400) &
        (leaders['Gap'] >= 0.15) &
        (leaders['CareerPrize'] >= 20000)
    ].copy()
    
    short_bsp = short_dom[short_dom['BSP'].notna() & (short_dom['BSP'] >= 2.0) & (short_dom['BSP'] <= 30)].copy()
    short_sp = short_dom[short_dom['SP'].notna() & (short_dom['SP'] >= 2.0) & (short_dom['SP'] <= 30)].copy()
    
    if len(short_bsp) > 0:
        short_bsp['ProfitBSP'] = short_bsp.apply(lambda x: (x['BSP'] - 1) if x['Position'] == '1' else -1, axis=1)
        sbsp_wins = short_bsp[short_bsp['Position'] == '1'].shape[0]
        sbsp_strike = (sbsp_wins / len(short_bsp)) * 100
        sbsp_profit = short_bsp['ProfitBSP'].sum()
        sbsp_roi = (sbsp_profit / len(short_bsp)) * 100
    else:
        sbsp_wins, sbsp_strike, sbsp_profit, sbsp_roi = 0, 0, 0, 0
        
    if len(short_sp) > 0:
        short_sp['ProfitSP'] = short_sp.apply(lambda x: (x['SP'] - 1) if x['Position'] == '1' else -1, axis=1)
        ssp_wins = short_sp[short_sp['Position'] == '1'].shape[0]
        ssp_strike = (ssp_wins / len(short_sp)) * 100
        ssp_profit = short_sp['ProfitSP'].sum()
        ssp_roi = (ssp_profit / len(short_sp)) * 100
    else:
        ssp_wins, ssp_strike, ssp_profit, ssp_roi = 0, 0, 0, 0
    
    print("\n" + "-"*60)
    print(f"{'Strategy':<25} | {'Bets':<6} | {'Strike %':<9} | {'Profit':<10} | {'ROI %':<8}")
    print("-"*60)
    print(f"{'Short Dominant (BSP)':<25} | {len(short_bsp):<6} | {sbsp_strike:<9.1f} | {sbsp_profit:<10.1f} | {sbsp_roi:<8.1f}")
    print(f"{'Short Dominant (SP)':<25} | {len(short_sp):<6} | {ssp_strike:<9.1f} | {ssp_profit:<10.1f} | {ssp_roi:<8.1f}")
    
    # Average Odds Comparison
    print("\n" + "-"*60)
    print("AVERAGE ODDS COMPARISON")
    print("-"*60)
    if len(short_bsp) > 0 and len(short_sp) > 0:
        print(f"Short Dominant Avg BSP: ${short_bsp['BSP'].mean():.2f}")
        print(f"Short Dominant Avg SP:  ${short_sp['SP'].mean():.2f}")
        print(f"BSP Premium: {((short_bsp['BSP'].mean() / short_sp['SP'].mean()) - 1) * 100:.1f}%")

if __name__ == "__main__":
    run_bsp_backtest()
