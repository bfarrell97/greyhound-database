"""
Final Backtest of GUI Strategy (2025 Optimized)
Replicates the exact logic currently implemented in the GUI.

Filters:
1. Distance <= 600m
2. Field Size >= 6
3. Min Market Odds >= $1.30
4. Market Overround <= 1.40 (Data sanity check)

Strategies:
1. Leader: PIR=1, Pace=1, Prize>=$30k
2. Top 3:  PIR=1, Pace<=3, Prize>=$30k
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
start_date = '2020-01-01'
end_date = '2025-12-09'
min_odds = 1.50
max_odds = 30.0
DB_PATH = 'greyhound_racing.db'

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_final_backtest():
    conn = sqlite3.connect(DB_PATH)
    
    progress("Fetching race data...")
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
        print(f"Error: {e}")
        return

    # Cleaning
    races_df['Box'] = pd.to_numeric(races_df['Box'], errors='coerce')
    races_df['CurrentOdds'] = pd.to_numeric(races_df['CurrentOdds'], errors='coerce')
    races_df['Distance'] = pd.to_numeric(races_df['Distance'], errors='coerce')
    races_df['CareerPrizeMoney'] = races_df['CareerPrizeMoney'].astype(str).str.replace(r'[$,]', '', regex=True)
    races_df['CareerPrizeMoney'] = pd.to_numeric(races_df['CareerPrizeMoney'], errors='coerce').fillna(0)
    races_df['MeetingDate'] = pd.to_datetime(races_df['MeetingDate'])
    races_df['RaceKey'] = races_df['MeetingID'].astype(str) + '_R' + races_df['RaceNumber'].astype(str)
    
    # --- FIELD SIZE CALCULATION (Raw) ---
    races_df['FieldSize'] = races_df.groupby('RaceKey')['GreyhoundID'].transform('count')

    progress(f"Loaded {len(races_df):,} entries")

    # Historical Stats
    unique_dogs = races_df['GreyhoundID'].unique()
    dogs_str = ",".join(map(str, unique_dogs))
    
    progress("Fetching historical metrics...")
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

    # Rolling Calculations
    # SPLIT
    split_hist = history_df.dropna(subset=['Split']).copy()
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['Split'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # PACE
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist['HistAvgPace'] = pace_hist.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # PRIZE (Running)
    prize_hist = history_df.sort_values(['GreyhoundID', 'MeetingDate']).copy()
    prize_hist['CumPrize'] = prize_hist.groupby('GreyhoundID')['PrizeMoney'].cumsum()
    prize_hist['RunningPrize'] = prize_hist.groupby('GreyhoundID')['CumPrize'].shift(1).fillna(0)

    # Merge
    progress("Merging metrics...")
    races_df = races_df.merge(
        split_hist.dropna(subset=['HistAvgSplit'])[['GreyhoundID', 'MeetingDate', 'HistAvgSplit']],
        on=['GreyhoundID', 'MeetingDate'], how='left'
    )
    races_df = races_df.merge(
        pace_hist.dropna(subset=['HistAvgPace'])[['GreyhoundID', 'MeetingDate', 'HistAvgPace']],
        on=['GreyhoundID', 'MeetingDate'], how='left'
    )
    races_df = races_df.merge(
        prize_hist[['GreyhoundID', 'MeetingDate', 'RunningPrize']],
        on=['GreyhoundID', 'MeetingDate'], how='left'
    )
    races_df['RunningPrize'] = races_df['RunningPrize'].fillna(0)

    # --- MARKET FILTERS ---
    races_df['ImpliedProb'] = 1.0 / races_df['CurrentOdds'].replace(0, np.nan)
    market_stats = races_df.groupby('RaceKey').agg({
        'ImpliedProb': 'sum',
        'CurrentOdds': 'min'
    }).rename(columns={'ImpliedProb': 'MarketOverround', 'CurrentOdds': 'MinMarketOdds'}).reset_index()
    
    races_df = races_df.merge(market_stats, on='RaceKey', how='left')
    
    # --- GUI FILTERS APPLIED HERE ---
    # 1. Distance <= 600
    # 2. Field Size >= 6
    # 3. Min Odds >= 1.30
    # 4. Overround <= 1.40 (Sanity)
    
    filters_mask = (
        (races_df['Distance'] <= 600) &
        (races_df['FieldSize'] >= 6) &
        (races_df['MinMarketOdds'] >= 1.30) &
        (races_df['MarketOverround'] <= 1.40)
    )
    
    df = races_df[filters_mask].copy()
    df = df.dropna(subset=['HistAvgSplit', 'HistAvgPace'])
    
    progress(f"Qualified Runners after ALL filters: {len(df):,}")
    
    # Strategy Logic
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
    df['PredictedPIR'] = df['HistAvgSplit'] + df['BoxAdj']
    
    df['PredictedPIRRank'] = df.groupby('RaceKey')['PredictedPIR'].rank(method='min')
    df['PaceRank'] = df.groupby('RaceKey')['HistAvgPace'].rank(method='min', ascending=True)
    
    df['IsPIRLeader'] = df['PredictedPIRRank'] == 1
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPaceTop3'] = df['PaceRank'] <= 3
    df['HasMoney'] = df['RunningPrize'] >= 30000
    df['InOddsRange'] = (df['CurrentOdds'] >= min_odds) & (df['CurrentOdds'] <= max_odds)
    
    # Staking
    def get_stake(odds):
        if odds < 3: return 0.5
        elif odds < 5: return 0.75
        elif odds < 10: return 1.0
        elif odds < 20: return 1.5
        else: return 2.0

    df['Stake'] = df['CurrentOdds'].apply(get_stake)
    df['Return'] = df.apply(lambda r: r['Stake'] * r['CurrentOdds'] if r['IsWinner'] else 0, axis=1)
    df['Profit'] = df['Return'] - df['Stake']
    
    # Define Strategies
    strategies = {
        'Leader Strategy': df['IsPIRLeader'] & df['IsPaceLeader'] & df['HasMoney'] & df['InOddsRange'],
        'Top 3 Strategy':  df['IsPIRLeader'] & df['IsPaceTop3']   & df['HasMoney'] & df['InOddsRange']
    }
    
    # Output
    print("\n" + "="*80)
    print("FINAL BACKTEST RESULTS (GUI STRATEGY 2025)")
    print("Filters: <=600m, >=6 Dogs, Fav>$1.30, Overround<=1.40")
    print("="*80)
    print(f"{'Strategy':<20} | {'Bets':<8} | {'Win %':<8} | {'Profit (u)':<12} | {'ROI':<8}")
    print("-" * 70)
    
    with open('results/final_gui_backtest.txt', 'w') as f:
        f.write("FINAL BACKTEST RESULTS (GUI STRATEGY 2025)\n")
        f.write("Filters: <=600m, >=6 Dogs, Fav>$1.30, Overround<=1.40\n")
        f.write("="*80 + "\n")
        f.write(f"{'Strategy':<20} | {'Bets':<8} | {'Win %':<8} | {'Profit (u)':<12} | {'ROI':<8}\n")
        f.write("-" * 70 + "\n")
        
        for name, mask in strategies.items():
            subset = df[mask]
            count = len(subset)
            if count > 0:
                wins = subset['IsWinner'].sum()
                win_rate = (wins / count) * 100
                profit = subset['Profit'].sum()
                total_stake = subset['Stake'].sum()
                roi = (profit / total_stake) * 100
            else:
                win_rate = 0
                profit = 0
                roi = 0
            
            line = f"{name:<20} | {count:<8} | {win_rate:<8.1f} | {profit:<12.2f} | {roi:<8.1f}%"
            print(line)
            f.write(line + "\n")
            
            # Save detailed CSV
            filename = f"results/final_bets_{name.split()[0].lower()}.csv"
            subset.to_csv(filename, index=False)
            print(f"  Saved detailed bets to {filename}")

    conn.close()

if __name__ == "__main__":
    run_final_backtest()
