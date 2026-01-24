"""
Test ROI Improvements for Lay Strategy
Experiments: Margin, RaceNumber, Track, DayOfWeek, FieldSize, OddsBands
Does NOT modify any live code.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

DB_PATH = 'greyhound_racing.db'
STAKE = 100
COMM = 0.05  # 5% commission

def load_data():
    """Load and prepare data (reused from optimize_lay_strategy.py)"""
    print("Loading Data...")
    
    try:
        with open('tier1_tracks.txt', 'r') as f:
            safe_tracks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("tier1_tracks.txt not found, using all tracks")
        safe_tracks = None
        
    conn = sqlite3.connect(DB_PATH)
    
    if safe_tracks:
        placeholders = ','.join('?' for _ in safe_tracks)
        track_filter = f"AND t.TrackName IN ({placeholders})"
        params = safe_tracks
    else:
        track_filter = ""
        params = []
    
    query = f"""
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        r.RaceNumber,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Split,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2021-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      {track_filter}
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    print(f"Loaded {len(df)} entries")
    
    # Parse and clean
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['RaceNumber'] = pd.to_numeric(df['RaceNumber'], errors='coerce').fillna(0).astype(int)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek  # 0=Mon, 6=Sun
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # Field Size
    df['FieldSize'] = df.groupby('RaceID')['GreyhoundID'].transform('count')
    
    # Benchmark Feature
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    return df.dropna(subset=['DogNormTimeAvg', 'Odds'])

def get_predictions(df):
    """Train model and get predictions (Walk-Forward: Train <2023, Test >=2023)"""
    print("Training Model...")
    
    train_mask = df['MeetingDate'] < '2023-01-01'
    test_mask = df['MeetingDate'] >= '2023-01-01'
    
    train = df[train_mask]
    test = df[test_mask].copy()
    
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, tree_method='hist', verbosity=0)
    model.fit(train[features], train['NormTime'])
    
    test['PredOverall'] = model.predict(test[features])
    test['PredRank'] = test.groupby('RaceID')['PredOverall'].rank(method='min')
    
    print(f"Test Set: {len(test)} entries ({test['RaceID'].nunique()} races)")
    return test

def calculate_lay_pnl(subset):
    """Calculate Lay P&L for a subset"""
    if len(subset) == 0:
        return {'Bets': 0, 'Strike': 0, 'Profit': 0, 'ROI': 0, 'Stable': False}
    
    wins_lay = (subset['IsWin'] == 0).sum()
    gross_profit = wins_lay * STAKE * (1-COMM)
    
    subset = subset.copy()
    subset['Liability'] = (subset['Odds'] - 1) * STAKE
    losses_lay = subset[subset['IsWin'] == 1]['Liability'].sum()
    
    net_profit = gross_profit - losses_lay
    total_risk = subset['Liability'].sum()
    
    roi = net_profit / total_risk * 100 if total_risk > 0 else 0
    strike = wins_lay / len(subset) * 100 if len(subset) > 0 else 0
    
    # Stability Check (Profit each year)
    subset['Year'] = subset['MeetingDate'].dt.year
    yearly_profit = []
    for y in [2023, 2024, 2025]:
        y_sub = subset[subset['Year'] == y]
        if len(y_sub) == 0:
            yearly_profit.append(0)
        else:
            y_w = (y_sub['IsWin'] == 0).sum()
            y_gross = y_w * STAKE * (1-COMM)
            y_loss = y_sub[y_sub['IsWin'] == 1]['Liability'].sum()
            yearly_profit.append(y_gross - y_loss)
    
    stable = all(p > 0 for p in yearly_profit)
    
    return {
        'Bets': len(subset),
        'Strike': strike,
        'Profit': net_profit,
        'ROI': roi,
        'Stable': stable,
        'YearlyProfit': yearly_profit
    }

def get_base_candidates(test_df, margin_threshold=0.1):
    """Get base Lay candidates (Rank 1, Margin > threshold)"""
    rank1s = test_df[test_df['PredRank'] == 1].copy()
    rank2s = test_df[test_df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    return candidates[candidates['Margin'] > margin_threshold].copy()

# =============================================================================
# EXPERIMENTS
# =============================================================================

def test_margin_thresholds(test_df):
    """Test different margin thresholds"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: MARGIN THRESHOLD")
    print("="*80)
    
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    
    print(f"\n{'Margin':<10} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 60)
    
    results = []
    for t in thresholds:
        pool = get_base_candidates(test_df, margin_threshold=t)
        pool = pool[(pool['Odds'] >= 1.5) & (pool['Odds'] <= 3.0)]  # Standard odds filter
        
        pnl = calculate_lay_pnl(pool)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f">{t:<9} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")
        results.append((t, pnl))
    
    return results

def test_race_number(test_df):
    """Test Early vs Middle vs Late races"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: RACE NUMBER (Early/Mid/Late)")
    print("="*80)
    
    pool = get_base_candidates(test_df, margin_threshold=0.1)
    pool = pool[(pool['Odds'] >= 1.5) & (pool['Odds'] <= 3.0)]
    
    race_groups = {
        'Early (R1-3)': pool[pool['RaceNumber'].isin([1,2,3])],
        'Mid (R4-7)': pool[pool['RaceNumber'].isin([4,5,6,7])],
        'Late (R8+)': pool[pool['RaceNumber'] >= 8],
        'All': pool
    }
    
    print(f"\n{'RaceGroup':<20} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 70)
    
    for name, subset in race_groups.items():
        pnl = calculate_lay_pnl(subset)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{name:<20} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")

def test_day_of_week(test_df):
    """Test Day of Week effect"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: DAY OF WEEK")
    print("="*80)
    
    pool = get_base_candidates(test_df, margin_threshold=0.1)
    pool = pool[(pool['Odds'] >= 1.5) & (pool['Odds'] <= 3.0)]
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    print(f"\n{'Day':<10} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 60)
    
    for i, name in enumerate(day_names):
        subset = pool[pool['DayOfWeek'] == i]
        pnl = calculate_lay_pnl(subset)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{name:<10} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")
    
    # Weekend vs Weekday
    print("\n-- Weekend vs Weekday --")
    weekend = pool[pool['DayOfWeek'].isin([5, 6])]
    weekday = pool[pool['DayOfWeek'].isin([0,1,2,3,4])]
    
    for name, subset in [('Weekend', weekend), ('Weekday', weekday)]:
        pnl = calculate_lay_pnl(subset)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{name:<10} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")

def test_field_size(test_df):
    """Test Field Size effect"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: FIELD SIZE")
    print("="*80)
    
    pool = get_base_candidates(test_df, margin_threshold=0.1)
    pool = pool[(pool['Odds'] >= 1.5) & (pool['Odds'] <= 3.0)]
    
    field_groups = {
        'Small (<=6)': pool[pool['FieldSize'] <= 6],
        'Medium (7)': pool[pool['FieldSize'] == 7],
        'Full (8)': pool[pool['FieldSize'] == 8],
        'All': pool
    }
    
    print(f"\n{'FieldSize':<15} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 65)
    
    for name, subset in field_groups.items():
        pnl = calculate_lay_pnl(subset)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{name:<15} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")

def test_odds_bands(test_df):
    """Test different Odds Bands"""
    print("\n" + "="*80)
    print("EXPERIMENT 5: ODDS BANDS")
    print("="*80)
    
    pool = get_base_candidates(test_df, margin_threshold=0.1)
    
    bands = [
        ('1.50-1.80', 1.50, 1.80),
        ('1.80-2.00', 1.80, 2.00),
        ('2.00-2.20', 2.00, 2.20),
        ('2.20-2.50', 2.20, 2.50),
        ('2.50-3.00', 2.50, 3.00),
        ('3.00-4.00', 3.00, 4.00),
        ('1.50-2.50 (Tight)', 1.50, 2.50),
        ('1.50-3.00 (Standard)', 1.50, 3.00),
    ]
    
    print(f"\n{'OddsBand':<20} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 70)
    
    for name, low, high in bands:
        subset = pool[(pool['Odds'] >= low) & (pool['Odds'] <= high)]
        pnl = calculate_lay_pnl(subset)
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{name:<20} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")

def test_top_tracks(test_df):
    """Find best performing tracks"""
    print("\n" + "="*80)
    print("EXPERIMENT 6: TRACK PERFORMANCE (Top 10)")
    print("="*80)
    
    pool = get_base_candidates(test_df, margin_threshold=0.1)
    pool = pool[(pool['Odds'] >= 1.5) & (pool['Odds'] <= 3.0)]
    
    track_results = []
    for track in pool['TrackName'].unique():
        subset = pool[pool['TrackName'] == track]
        if len(subset) >= 50:  # Min sample
            pnl = calculate_lay_pnl(subset)
            track_results.append((track, pnl))
    
    # Sort by ROI
    track_results.sort(key=lambda x: x[1]['ROI'], reverse=True)
    
    print(f"\n{'Track':<25} {'Bets':<8} {'Strike%':<10} {'Profit':<12} {'ROI%':<8} {'Stable':<8}")
    print("-" * 75)
    
    for track, pnl in track_results[:10]:
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{track:<25} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")
    
    print("\n-- Worst 5 Tracks --")
    for track, pnl in track_results[-5:]:
        stable_str = "YES" if pnl['Stable'] else "NO"
        print(f"{track:<25} {pnl['Bets']:<8} {pnl['Strike']:<10.1f} ${pnl['Profit']:<11.0f} {pnl['ROI']:<8.1f} {stable_str:<8}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LAY STRATEGY ROI IMPROVEMENT TESTS")
    print("="*80)
    
    df = load_data()
    test_df = get_predictions(df)
    
    # Run all experiments
    test_margin_thresholds(test_df)
    test_race_number(test_df)
    test_day_of_week(test_df)
    test_field_size(test_df)
    test_odds_bands(test_df)
    test_top_tracks(test_df)
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
