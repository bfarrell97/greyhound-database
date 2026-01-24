"""
Backtest Variations: Leader vs Top 3 with Fav < $1.30 Filter
Based on backtest_production_strategy_adam.py
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

def run_variation_analysis():
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
    
    # Calculate Field Size BEFORE filtering (count of runners per race)
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
    split_hist = history_df.dropna(subset=['Split']).copy()
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['Split'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist['HistAvgPace'] = pace_hist.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
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

    # Market Filters
    races_df['ImpliedProb'] = 1.0 / races_df['CurrentOdds'].replace(0, np.nan)
    market_stats = races_df.groupby('RaceKey').agg({
        'ImpliedProb': 'sum',
        'CurrentOdds': 'min'
    }).rename(columns={'ImpliedProb': 'MarketOverround', 'CurrentOdds': 'MinMarketOdds'}).reset_index()
    
    races_df = races_df.merge(market_stats, on='RaceKey', how='left')
    
    # Base Filters (Overround <= 1.40) & Qualifiers
    df = races_df[races_df['MarketOverround'] <= 1.40].copy()
    df = df.dropna(subset=['HistAvgSplit', 'HistAvgPace'])
    
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
    
    # --- NEW FILTER: Exclude races with heavy favorite (< $1.30) ---
    df['MinMarketOdds'] = df.groupby('RaceKey')['CurrentOdds'].transform('min')

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

    # --- DEFINE VARIATIONS ---
    strategies = {
        'Leader Strategy': df['IsPIRLeader'] & df['IsPaceLeader'] & df['HasMoney'] & df['InOddsRange'],
        'Top 3 Strategy':  df['IsPIRLeader'] & df['IsPaceTop3']   & df['HasMoney'] & df['InOddsRange']
    }
    
    scenarios = {
        '1. Baseline': lambda d: pd.Series([True]*len(d), index=d.index),
        '2. No Small Fields (< 6 Dogs)': lambda d: d['FieldSize'] >= 6
    }
    
    progress("\nCALCULATING VARIATIONS:")
    
    # Output File
    with open('results/field_size_report.txt', 'w') as f:
        f.write(f"\nRESULTS REPORT [Field Size Test] [{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n")
        f.write("=" * 80 + "\n")
        
        for strat_name, strat_mask in strategies.items():
            print(f"\n--- {strat_name} ---")
            print(f"{'Scenario':<30} | {'Bets':<6} | {'Win %':<6} | {'Profit':<10} | {'ROI':<8}")
            print("-" * 75)
            
            f.write(f"\n--- {strat_name} ---\n")
            f.write(f"{'Scenario':<30} | {'Bets':<6} | {'Win %':<6} | {'Profit':<10} | {'ROI':<8}\n")
            f.write("-" * 80 + "\n")
            
            strat_df = df[strat_mask].copy()
            
            for scen_name, scen_func in scenarios.items():
                mask = scen_func(strat_df)
                subset = strat_df[mask]
                
                bets_count = len(subset)
                if bets_count > 0:
                    win_rate = (subset['IsWinner'].sum() / bets_count) * 100
                    profit = subset['Profit'].sum()
                    roi = (profit / subset['Stake'].sum()) * 100
                else:
                    win_rate = 0
                    profit = 0
                    roi = 0
                
                line = f"{scen_name:<30} | {bets_count:<6} | {win_rate:<6.1f} | {profit:<10.2f} | {roi:<7.1f}%"
                print(line)
                f.write(line + "\n")
    
    conn.close()

if __name__ == "__main__":
    run_variation_analysis()
