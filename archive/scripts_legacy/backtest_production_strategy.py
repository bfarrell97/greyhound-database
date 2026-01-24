"""
Fast Backtest for Production Strategy (Long Term: 2020-2025)
Runs both:
1. PIR + Pace Leader + $30k
2. PIR + Pace Top 3 + $30k
Uses bulk data loading.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

# Configuration
configs = [
    {'mode': 'leader', 'name': 'PIR + Pace Leader + $30k'},
    {'mode': 'top3', 'name': 'PIR + Pace Top 3 + $30k'}
]

# Date Range (Long Term)
start_date = '2020-01-01'
end_date = '2025-12-09'
min_odds = 1.50
max_odds = 30.0

DB_PATH = 'greyhound_racing.db'

def run_backtest(config):
    strategy_mode = config['mode']
    strategy_name = config['name']
    
    conn = sqlite3.connect(DB_PATH)
    
    progress("=" * 100)
    progress(f"RUNNING BACKTEST: {strategy_name}")
    progress(f"Period: {start_date} to {end_date}")
    progress("=" * 100)
    
    # 1. Fetch Qualification Races
    progress("Fetching race candidates...")
    races_query = f"""
    SELECT 
        rm.MeetingID,
        rm.MeetingDate,
        t.TrackName,
        r.RaceNumber,
        r.RaceTime,
        r.Distance,
        ge.GreyhoundID,
        g.GreyhoundName,
        ge.Box,
        ge.StartingPrice as CurrentOdds,
        ge.CareerPrizeMoney,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '{start_date}' 
      AND rm.MeetingDate <= '{end_date}'
      AND ge.StartingPrice IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.Box IS NOT NULL
    """
    try:
        races_df = pd.read_sql_query(races_query, conn)
    except Exception as e:
        print(f"Error querying races: {e}")
        return

    # Numeric conversions
    races_df['Box'] = pd.to_numeric(races_df['Box'], errors='coerce')
    races_df['CurrentOdds'] = pd.to_numeric(races_df['CurrentOdds'], errors='coerce')

    # Clean CareerPrizeMoney
    races_df['CareerPrizeMoney'] = races_df['CareerPrizeMoney'].astype(str).str.replace(r'[$,]', '', regex=True)
    races_df['CareerPrizeMoney'] = pd.to_numeric(races_df['CareerPrizeMoney'], errors='coerce').fillna(0)
    
    progress(f"  Found {len(races_df):,} entries")

    # Filter out races with bad SP data (overround outside 90-130%)
    races_df['RaceKey'] = races_df['MeetingID'].astype(str) + '_R' + races_df['RaceNumber'].astype(str)
    races_df['ImpliedProb'] = 1 / races_df['CurrentOdds']
    race_overround = races_df.groupby('RaceKey')['ImpliedProb'].transform('sum') * 100
    races_df['Overround'] = race_overround
    
    before_filter = len(races_df)
    races_df = races_df[(races_df['Overround'] >= 90) & (races_df['Overround'] <= 130)]
    progress(f"  After overround filter (90-130%): {len(races_df):,} entries ({before_filter - len(races_df):,} excluded)")

    # 2. History
    unique_dogs = races_df['GreyhoundID'].unique()
    dogs_str = ",".join(map(str, unique_dogs))
    
    progress("Fetching historical data (this may take a moment)...")
    history_query = f"""
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        ge.Split,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney,
        (ge.FinishTimeBenchmarkLengths + COALESCE(rm.MeetingAvgBenchmarkLengths, 0)) as TotalPace
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.GreyhoundID IN ({dogs_str})
      AND rm.MeetingDate < '{end_date}'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate ASC
    """
    history_df = pd.read_sql_query(history_query, conn)
    history_df['MeetingDate'] = pd.to_datetime(history_df['MeetingDate'])
    progress(f"  Loaded {len(history_df):,} history points")
    
    # 3. Rolling Metrics
    progress("Calculating metrics...")
    
    split_hist = history_df.dropna(subset=['Split']).copy()
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['Split'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist['HistAvgPace'] = history_df.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )

    # Rolling Prize Money (Fix Leakage)
    prize_hist = history_df.sort_values(['GreyhoundID', 'MeetingDate']).copy()
    prize_hist['CumPrize'] = prize_hist.groupby('GreyhoundID')['PrizeMoney'].cumsum()
    prize_hist['RunningPrize'] = prize_hist.groupby('GreyhoundID')['CumPrize'].shift(1).fillna(0)

    races_df['MeetingDate'] = pd.to_datetime(races_df['MeetingDate'])
    
    # Merges
    races_df = races_df.merge(
        split_hist[['GreyhoundID', 'MeetingDate', 'HistAvgSplit']], 
        on=['GreyhoundID', 'MeetingDate'], 
        how='left'
    )
    
    races_df = races_df.merge(
        pace_hist[['GreyhoundID', 'MeetingDate', 'HistAvgPace']], 
        on=['GreyhoundID', 'MeetingDate'], 
        how='left'
    )
    
    races_df = races_df.merge(
        prize_hist[['GreyhoundID', 'MeetingDate', 'RunningPrize']], 
        on=['GreyhoundID', 'MeetingDate'], 
        how='left'
    )
    races_df['RunningPrize'] = races_df['RunningPrize'].fillna(0)
    
    valid_entries = races_df.dropna(subset=['HistAvgSplit', 'HistAvgPace']).copy()
    progress(f"  Qualified entries: {len(valid_entries):,}")
    
    if len(valid_entries) == 0:
        print("No qualified entries.")
        conn.close()
        return

    # 4. Strategy Logic
    df = valid_entries
    
    # Box Adj
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
    df['PredictedPIR'] = df['HistAvgSplit'] + df['BoxAdj']
    
    # Rank
    df['RaceKey'] = df['MeetingID'].astype(str) + '_R' + df['RaceNumber'].astype(str)
    df['PredictedPIRRank'] = df.groupby('RaceKey')['PredictedPIR'].rank(method='min')
    df['PaceRank'] = df.groupby('RaceKey')['HistAvgPace'].rank(method='min', ascending=True)
    
    # Filters
    df['IsPIRLeader'] = df['PredictedPIRRank'] == 1
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPaceTop3'] = df['PaceRank'] <= 3
    df['IsPaceTop3'] = df['PaceRank'] <= 3
    df['HasMoney'] = df['RunningPrize'] >= 30000
    df['InOddsRange'] = (df['CurrentOdds'] >= min_odds) & (df['CurrentOdds'] <= max_odds)
    df['InOddsRange'] = (df['CurrentOdds'] >= min_odds) & (df['CurrentOdds'] <= max_odds)
    
    if strategy_mode == 'leader':
        bets = df[df['IsPIRLeader'] & df['IsPaceLeader'] & df['HasMoney'] & df['InOddsRange']].copy()
    else:
        bets = df[df['IsPIRLeader'] & df['IsPaceTop3'] & df['HasMoney'] & df['InOddsRange']].copy()
        
    progress(f"  Bets Found: {len(bets)}")
    
    if len(bets) == 0:
        print("No bets found.")
        conn.close()
        return

    # Staking
    def get_stake(odds):
        if odds < 3: return 0.5
        elif odds < 5: return 0.75
        elif odds < 10: return 1.0
        elif odds < 20: return 1.5
        else: return 2.0

    bets['Stake'] = bets['CurrentOdds'].apply(get_stake)
    bets['Return'] = bets.apply(lambda row: row['Stake'] * row['CurrentOdds'] if row['IsWinner'] else 0, axis=1)
    bets['Profit'] = bets['Return'] - bets['Stake']
    
    # Flat Staking
    bets['FlatProfit'] = bets.apply(lambda row: (row['CurrentOdds'] - 1) if row['IsWinner'] else -1, axis=1)

    # Save
    filename = f"results/backtest_longterm_{strategy_mode}.csv"
    bets.to_csv(filename, index=False)
    
    # Report
    total_profit = bets['Profit'].sum()
    flat_profit = bets['FlatProfit'].sum()
    roi = (total_profit / bets['Stake'].sum()) * 100
    flat_roi = (flat_profit / len(bets)) * 100
    
    print(f"\nRESULTS [{strategy_name}]:")
    print(f"  Bets: {len(bets)}")
    print(f"  Strike Rate: {(bets['IsWinner'].sum()/len(bets)*100):.2f}%")
    print(f"  Profit (Inverse): {total_profit:+.2f}u (ROI: {roi:+.2f}%)")
    print(f"  Profit (Flat):    {flat_profit:+.2f}u (ROI: {flat_roi:+.2f}%)")
    print("-" * 50)
    
    conn.close()
    return bets

if __name__ == "__main__":
    for config in configs:
        run_backtest(config)
