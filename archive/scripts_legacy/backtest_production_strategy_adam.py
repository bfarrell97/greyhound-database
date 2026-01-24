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
import sqlite3


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
    races_df['CurrentOdds'] = pd.to_numeric(
        races_df['CurrentOdds'], errors='coerce')

    # Clean CareerPrizeMoney
    races_df['CareerPrizeMoney'] = races_df['CareerPrizeMoney'].astype(
        str).str.replace(r'[$,]', '', regex=True)
    races_df['CareerPrizeMoney'] = pd.to_numeric(
        races_df['CareerPrizeMoney'], errors='coerce').fillna(0)

    progress(f"  Found {len(races_df):,} entries")

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

    # Compute rolling averages on history_df (shift to avoid lookahead)
    # SPLIT: Drop null splits first so we get "Last 5 recorded splits"
    split_hist = history_df.dropna(subset=['Split']).copy()
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['Split'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )

    # PACE: Drop null pace first
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist['HistAvgPace'] = pace_hist.groupby('GreyhoundID')['TotalPace'].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=5).mean()
    )

    # Rolling Prize Money (Fix Leakage)
    prize_hist = history_df.sort_values(['GreyhoundID', 'MeetingDate']).copy()
    prize_hist['CumPrize'] = prize_hist.groupby('GreyhoundID')['PrizeMoney'].cumsum()
    prize_hist['RunningPrize'] = prize_hist.groupby('GreyhoundID')['CumPrize'].shift(1).fillna(0)

    # Filter to only rows with the metric
    split_hist = split_hist.dropna(subset=['HistAvgSplit'])[['GreyhoundID', 'MeetingDate', 'HistAvgSplit']]
    pace_hist = pace_hist.dropna(subset=['HistAvgPace'])[['GreyhoundID', 'MeetingDate', 'HistAvgPace']]

    races_df['MeetingDate'] = pd.to_datetime(races_df['MeetingDate'])

    # create RaceKey early so we can calculate market-level filters
    races_df['RaceKey'] = races_df['MeetingID'].astype(
        str) + '_R' + races_df['RaceNumber'].astype(str)

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

    progress(f"  Entries before qualification: {len(races_df):,}")

    # Add simple market overround filter (sum of implied probs) to exclude poor markets
    races_df['ImpliedProb'] = 1.0 / races_df['CurrentOdds'].replace(0, np.nan)
    overround = races_df.groupby(
        'RaceKey')['ImpliedProb'].sum().reset_index(name='MarketOverround')
    races_df = races_df.merge(overround, on='RaceKey', how='left')

    # Diagnostics: explain why markets are filtered
    total_racekeys = races_df['RaceKey'].nunique()
    na_odds = races_df['CurrentOdds'].isna().sum()
    zero_odds = (races_df['CurrentOdds'] == 0).sum()
    neg_odds = (races_df['CurrentOdds'] < 0).sum()
    progress(
        f"  Market overround diagnostics: {total_racekeys:,} unique races")
    progress(
        f"    CurrentOdds null: {na_odds:,}, zeros: {zero_odds:,}, negative: {neg_odds:,}")

    # Show top races by overround (highest implied probability sums)
    top_over = overround.sort_values(
        'MarketOverround', ascending=False).head(20)
    progress("    Top 20 races by MarketOverround:")
    print(top_over.to_string(index=False))

    # Show count of races that will be removed by the overround threshold
    removed_races = overround[overround['MarketOverround'] > 1.40]
    progress(f"    Races with MarketOverround > 1.40: {len(removed_races):,}")
    if len(removed_races) > 0:
        sample_removed = removed_races.head(10)['RaceKey'].tolist()
        progress("    Sample removed RaceKeys and their overround:")
        print(removed_races.head(10).to_string(index=False))

        # Print example entries for one removed race to inspect odds
        example_rk = sample_removed[0]
        progress(f"    Example removed race entries for {example_rk}:")
        print(races_df[races_df['RaceKey'] == example_rk][[
              'GreyhoundID', 'GreyhoundName', 'CurrentOdds', 'ImpliedProb']].to_string(index=False))

    before_or = len(races_df)
    races_df = races_df[races_df['MarketOverround'] <= 1.40]
    progress(
        f"  Entries after market overround filter: {len(races_df):,} (removed {before_or - len(races_df):,})")

    # Diagnostics for why rows will be dropped by requiring both HistAvgSplit and HistAvgPace
    progress("  Diagnostics for qualification drop:")
    try:
        progress(f"    split_hist rows (computed): {len(split_hist):,}")
    except Exception:
        pass
    try:
        progress(f"    pace_hist rows (computed): {len(pace_hist):,}")
    except Exception:
        pass

    split_nonnull = races_df['HistAvgSplit'].notna().sum()
    pace_nonnull = races_df['HistAvgPace'].notna().sum()
    both_nonnull = races_df.dropna(
        subset=['HistAvgSplit', 'HistAvgPace']).shape[0]
    progress(f"    Rows with HistAvgSplit: {split_nonnull:,}")
    progress(f"    Rows with HistAvgPace: {pace_nonnull:,}")
    progress(f"    Rows with both: {both_nonnull:,}")

    # Unique dog counts
    dogs_with_split = races_df.loc[races_df['HistAvgSplit'].notna(
    ), 'GreyhoundID'].nunique()
    dogs_with_pace = races_df.loc[races_df['HistAvgPace'].notna(
    ), 'GreyhoundID'].nunique()
    dogs_with_both = races_df.dropna(subset=['HistAvgSplit', 'HistAvgPace'])[
        'GreyhoundID'].nunique()
    progress(f"    Dogs with HistAvgSplit: {dogs_with_split:,}")
    progress(f"    Dogs with HistAvgPace: {dogs_with_pace:,}")
    progress(f"    Dogs with both: {dogs_with_both:,}")

    # Show samples of rows missing one or both metrics
    missing_both = races_df[races_df['HistAvgSplit'].isna(
    ) & races_df['HistAvgPace'].isna()]
    missing_split_only = races_df[races_df['HistAvgSplit'].isna(
    ) & races_df['HistAvgPace'].notna()]
    missing_pace_only = races_df[races_df['HistAvgPace'].isna(
    ) & races_df['HistAvgSplit'].notna()]

    progress(
        f"    Rows missing both: {len(missing_both):,}, missing split only: {len(missing_split_only):,}, missing pace only: {len(missing_pace_only):,}")
    if len(missing_both) > 0:
        progress("    Sample rows missing both metrics:")
        print(missing_both[['RaceKey', 'GreyhoundID', 'GreyhoundName',
              'CurrentOdds', 'MeetingDate']].head(10).to_string(index=False))
    if len(missing_split_only) > 0:
        progress("    Sample rows missing HistAvgSplit only:")
        print(missing_split_only[['RaceKey', 'GreyhoundID', 'GreyhoundName',
              'CurrentOdds', 'HistAvgPace', 'MeetingDate']].head(10).to_string(index=False))
    if len(missing_pace_only) > 0:
        progress("    Sample rows missing HistAvgPace only:")
        print(missing_pace_only[['RaceKey', 'GreyhoundID', 'GreyhoundName',
              'CurrentOdds', 'HistAvgSplit', 'MeetingDate']].head(10).to_string(index=False))

    valid_entries = races_df.dropna(
        subset=['HistAvgSplit', 'HistAvgPace']).copy()
    progress(
        f"  Qualified entries: {len(valid_entries):,} (dropped {len(races_df) - len(valid_entries):,})")

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
    df['RaceKey'] = df['MeetingID'].astype(
        str) + '_R' + df['RaceNumber'].astype(str)
    df['PredictedPIRRank'] = df.groupby(
        'RaceKey')['PredictedPIR'].rank(method='min')
    df['PaceRank'] = df.groupby('RaceKey')['HistAvgPace'].rank(
        method='min', ascending=True)

    # Filters
    df['IsPIRLeader'] = df['PredictedPIRRank'] == 1
    df['IsPaceLeader'] = df['PaceRank'] == 1
    df['IsPaceTop3'] = df['PaceRank'] <= 3
    df['HasMoney'] = df['RunningPrize'] >= 30000
    df['InOddsRange'] = (df['CurrentOdds'] >= min_odds) & (
        df['CurrentOdds'] <= max_odds)

    if strategy_mode == 'leader':
        bets = df[df['IsPIRLeader'] & df['IsPaceLeader']
                  & df['HasMoney'] & df['InOddsRange']].copy()
    else:
        bets = df[df['IsPIRLeader'] & df['IsPaceTop3']
                  & df['HasMoney'] & df['InOddsRange']].copy()

    progress(f"  Bets Found: {len(bets)}")

    if len(bets) == 0:
        print("No bets found.")
        conn.close()
        return

    # Check for multiple qualifying bets per race (ties etc.)
    dup_counts = bets.groupby('RaceKey').size().reset_index(name='count')
    multi_races = dup_counts[dup_counts['count'] > 1]
    if len(multi_races) > 0:
        progress(
            f"  WARNING: {len(multi_races)} races have >1 qualifying bet (ties or duplicates). Showing up to 10:")
        print(multi_races.head(10).to_string(index=False))

    # Staking
    def get_stake(odds):
        if odds < 3:
            return 0.5
        elif odds < 5:
            return 0.75
        elif odds < 10:
            return 1.0
        elif odds < 20:
            return 1.5
        else:
            return 2.0

    bets['Stake'] = bets['CurrentOdds'].apply(get_stake)
    bets['Return'] = bets.apply(
        lambda row: row['Stake'] * row['CurrentOdds'] if row['IsWinner'] else 0, axis=1)
    bets['Profit'] = bets['Return'] - bets['Stake']

    # Flat Staking
    bets['FlatProfit'] = bets.apply(lambda row: (
        row['CurrentOdds'] - 1) if row['IsWinner'] else -1, axis=1)

    # Debug: show sample bets and stake distribution
    progress("Displaying sample bets and stakes:")
    try:
        print(bets[['RaceKey', 'GreyhoundID', 'GreyhoundName', 'CurrentOdds',
              'Stake', 'IsWinner']].head(20).to_string(index=False))
    except Exception:
        print(bets.head(20).to_string(index=False))

    dup_counts_after = bets.groupby('RaceKey').size().reset_index(name='count')
    multi_after = dup_counts_after[dup_counts_after['count'] > 1]
    if len(multi_after) > 0:
        progress(
            f"  WARNING: {len(multi_after)} races still have >1 bet after staking. First 10:")
        print(multi_after.head(10).to_string(index=False))

    # Save
    filename_db = f"results/backtest_longterm_{strategy_mode}.db"
    conn = sqlite3.connect(filename_db)
    bets.to_sql('BacktestResults', conn, if_exists='replace', index=False)

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
