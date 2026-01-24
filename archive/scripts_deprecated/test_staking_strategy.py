"""
Test Tiered Staking Strategy vs Flat Staking
Based on insights from ROI improvement tests:
- Late Races (R8+) = +1 point
- Weekend (Sat/Sun) = +1 point
- Tight Odds (1.50-2.00) = +1 point
- Small/Medium Field (<=7) = +1 point
- Top Track = +1 point
- Margin > 0.15 = +1 point

Tiered Staking: Score 0-2 = $50, Score 3-4 = $100, Score 5+ = $150
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'
COMM = 0.05  # 5% commission

# Top performing tracks from experiment
TOP_TRACKS = [
    'Murray Bridge (MBR)', 'The Meadows', 'Wentworth Park', 'Grafton',
    'Gosford', 'The Gardens', 'Gawler', 'Ballarat', 'Dapto'
]

# Worst tracks to avoid
AVOID_TRACKS = ['Meadows (MEP)', 'Sale']

def load_data():
    """Load and prepare data"""
    print("Loading Data...")
    
    try:
        with open('tier1_tracks.txt', 'r') as f:
            safe_tracks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
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
        ge.Position,
        ge.StartingPrice,
        ge.BSP
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
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['RaceNumber'] = pd.to_numeric(df['RaceNumber'], errors='coerce').fillna(0).astype(int)
    df['IsWin'] = (df['Position'] == '1').astype(int)
    df['DayOfWeek'] = df['MeetingDate'].dt.dayofweek
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
        
    df['SP_Clean'] = df['StartingPrice'].apply(parse_price)
    # Use BSP if available, else SP
    df['Odds'] = df['BSP'].fillna(df['SP_Clean'])
    
    # Store flag for analysis
    df['PriceSource'] = np.where(df['BSP'].notna(), 'BSP', 'SP')
    
    df['FieldSize'] = df.groupby('RaceID')['GreyhoundID'].transform('count')
    
    bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    bench.columns = ['TrackName', 'Distance', 'MedianTime']
    df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['MedianTime']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    print(f"Loaded {len(df)} entries")
    return df.dropna(subset=['DogNormTimeAvg', 'Odds'])

def get_predictions(df):
    """Load pre-trained model and get predictions"""
    import joblib
    import os
    
    MODEL_PATH = 'src/models/lay_model.pkl'
    
    # Only use test data (2023+)
    test_mask = df['MeetingDate'] >= '2023-01-01'
    test = df[test_mask].copy()
    
    # Load pre-trained model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        loaded = joblib.load(MODEL_PATH)
        # Handle dict or raw model
        if isinstance(loaded, dict) and 'model' in loaded:
            model = loaded['model']
        else:
            model = loaded
            
        print(f"Loaded model type: {type(model)}")
    else:
        print(f"Model not found at {MODEL_PATH}, training new model...")
        train_mask = df['MeetingDate'] < '2023-01-01'
        train = df[train_mask]
        features = ['DogNormTimeAvg', 'Box', 'Distance']
        model = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=100, n_jobs=-1, tree_method='hist', verbosity=0)
        model.fit(train[features], train['NormTime'])
    
    features = ['DogNormTimeAvg', 'Box', 'Distance']
    test['PredOverall'] = model.predict(test[features])
    test['PredRank'] = test.groupby('RaceID')['PredOverall'].rank(method='min')
    
    print(f"Test Set: {len(test)} entries")
    return test

def get_candidates(test_df, margin_threshold=0.1):
    """Get Lay candidates with margin"""
    rank1s = test_df[test_df['PredRank'] == 1].copy()
    rank2s = test_df[test_df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    # Base filter
    pool = candidates[
        (candidates['Margin'] > margin_threshold) &
        (candidates['Odds'] >= 1.50) &
        (candidates['Odds'] <= 3.00) &
        (~candidates['TrackName'].isin(AVOID_TRACKS))
    ].copy()
    
    return pool

def score_bet(row):
    """Score a bet based on positive factors"""
    score = 0
    
    # Late Race (R8+)
    if row['RaceNumber'] >= 8:
        score += 1
    
    # Weekend (Sat=5, Sun=6)
    if row['DayOfWeek'] in [5, 6]:
        score += 1
    
    # Tight Odds (1.50-2.00)
    if 1.50 <= row['Odds'] <= 2.00:
        score += 1
    
    # Small/Medium Field (<=7)
    if row['FieldSize'] <= 7:
        score += 1
    
    # Top Track
    if row['TrackName'] in TOP_TRACKS:
        score += 1
    
    # High Margin (>0.15)
    if row['Margin'] > 0.15:
        score += 1
    
    return score

def get_tiered_stake(score):
    """Get stake based on confidence score"""
    if score <= 2:
        return 50   # Low confidence
    elif score <= 4:
        return 100  # Medium confidence
    else:
        return 150  # High confidence

def simulate_strategy(pool, use_tiered=True):
    """Simulate betting with either tiered or flat staking"""
    FLAT_STAKE = 100
    
    pool = pool.copy()
    pool['Score'] = pool.apply(score_bet, axis=1)
    
    if use_tiered:
        pool['Stake'] = pool['Score'].apply(get_tiered_stake)
    else:
        pool['Stake'] = FLAT_STAKE
    
    pool['Liability'] = (pool['Odds'] - 1) * pool['Stake']
    
    # Calculate P&L
    # Win Lay = Dog loses, we collect stake * (1 - comm)
    # Lose Lay = Dog wins, we pay liability
    
    pool['Profit'] = pool.apply(
        lambda r: r['Stake'] * (1 - COMM) if r['IsWin'] == 0 else -r['Liability'],
        axis=1
    )
    
    total_profit = pool['Profit'].sum()
    total_risk = pool['Liability'].sum()
    total_stake = pool['Stake'].sum()
    
    roi_risk = total_profit / total_risk * 100 if total_risk > 0 else 0
    roi_stake = total_profit / total_stake * 100 if total_stake > 0 else 0
    
    wins = (pool['IsWin'] == 0).sum()
    losses = (pool['IsWin'] == 1).sum()
    strike = wins / len(pool) * 100 if len(pool) > 0 else 0
    
    # Yearly breakdown
    pool['Year'] = pool['MeetingDate'].dt.year
    yearly = pool.groupby('Year')['Profit'].sum().to_dict()
    
    return {
        'Bets': len(pool),
        'Wins': wins,
        'Losses': losses,
        'Strike': strike,
        'TotalProfit': total_profit,
        'TotalRisk': total_risk,
        'TotalStake': total_stake,
        'ROI_Risk': roi_risk,
        'ROI_Stake': roi_stake,
        'Yearly': yearly,
        'ByScore': pool.groupby('Score')['Profit'].agg(['count', 'sum']).to_dict()
    }

def main():
    print("="*80)
    print("STAKING STRATEGY COMPARISON")
    print("="*80)
    
    df = load_data()
    test_df = get_predictions(df)
    pool = get_candidates(test_df)
    
    print(f"\nTotal Candidates: {len(pool)}")
    
    # Score each bet
    pool = pool.copy()
    pool['Score'] = pool.apply(score_bet, axis=1)
    
    # Score Distribution
    score_dist = pool['Score'].value_counts().sort_index()
    print("\n--- Score Distribution ---")
    for score, count in score_dist.items():
        print(f"Score {score}: {count:>5} bets")
    
    # Define Staking Strategies
    strategies = {
        'Flat $100': lambda s: 100,
        'Original Tiered': lambda s: 50 if s <= 2 else (100 if s <= 4 else 150),
        'Skip Low (3+)': lambda s: 0 if s <= 2 else (100 if s <= 4 else 150),
        'Aggressive (3+)': lambda s: 0 if s <= 2 else (100 if s <= 4 else 200),
        'Elite Only (5+)': lambda s: 0 if s < 5 else 200,
        'Super Elite (6)': lambda s: 0 if s < 6 else 300,
    }
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    results = []
    
    for name, stake_fn in strategies.items():
        pool_copy = pool.copy()
        pool_copy['Stake'] = pool_copy['Score'].apply(stake_fn)
        
        # Filter out $0 stakes
        betting_pool = pool_copy[pool_copy['Stake'] > 0].copy()
        
        if len(betting_pool) == 0:
            continue
        
        betting_pool['Liability'] = (betting_pool['Odds'] - 1) * betting_pool['Stake']
        betting_pool['Profit'] = betting_pool.apply(
            lambda r: r['Stake'] * (1 - COMM) if r['IsWin'] == 0 else -r['Liability'],
            axis=1
        )
        
        total_profit = betting_pool['Profit'].sum()
        total_stake = betting_pool['Stake'].sum()
        total_risk = betting_pool['Liability'].sum()
        
        wins = (betting_pool['IsWin'] == 0).sum()
        strike = wins / len(betting_pool) * 100
        
        roi_stake = total_profit / total_stake * 100 if total_stake > 0 else 0
        roi_risk = total_profit / total_risk * 100 if total_risk > 0 else 0
        
        # Calculate Max Drawdown
        betting_pool = betting_pool.sort_values('MeetingDate')
        betting_pool['CumProfit'] = betting_pool['Profit'].cumsum()
        betting_pool['CumMax'] = betting_pool['CumProfit'].cummax()
        betting_pool['Drawdown'] = betting_pool['CumProfit'] - betting_pool['CumMax']
        max_drawdown = betting_pool['Drawdown'].min()  # Most negative value
        
        # Yearly stability
        betting_pool['Year'] = betting_pool['MeetingDate'].dt.year
        yearly = betting_pool.groupby('Year')['Profit'].sum()
        stable = all(p > 0 for p in yearly.values)
        
        results.append({
            'Strategy': name,
            'Bets': len(betting_pool),
            'Strike': strike,
            'TotalStake': total_stake,
            'TotalProfit': total_profit,
            'ROI_Stake': roi_stake,
            'ROI_Risk': roi_risk,
            'MaxDrawdown': max_drawdown,
            'Stable': stable,
            'Yearly': yearly.to_dict()
        })
    
    # Print Results Table
    print(f"\n{'Strategy':<20} {'Bets':<7} {'Strike':<7} {'Profit':<11} {'ROI%':<7} {'MaxDD':<12} {'Stable':<6}")
    print("-" * 80)
    
    for r in results:
        stable_str = "YES" if r['Stable'] else "NO"
        print(f"{r['Strategy']:<20} {r['Bets']:<7} {r['Strike']:.1f}%{'':<2} ${r['TotalProfit']:>8,.0f} {r['ROI_Stake']:.1f}%{'':<2} ${r['MaxDrawdown']:>9,.0f} {stable_str:<6}")
    
    # Yearly Breakdown
    print("\n--- Yearly Profit by Strategy ---")
    years = [2023, 2024, 2025]
    header = f"{'Strategy':<20}"
    for y in years:
        header += f" {y:<12}"
    print(header)
    print("-" * 60)
    
    for r in results:
        line = f"{r['Strategy']:<20}"
        for y in years:
            p = r['Yearly'].get(y, 0)
            line += f" ${p:>10,.0f}"
        print(line)
    
    # Best Strategy Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    best_profit = max(results, key=lambda x: x['TotalProfit'])
    best_roi = max(results, key=lambda x: x['ROI_Stake'])
    
    print(f"\nBest Total Profit: {best_profit['Strategy']} (${best_profit['TotalProfit']:,.0f})")
    print(f"Best ROI:          {best_roi['Strategy']} ({best_roi['ROI_Stake']:.1f}%)")
    
    # Profit per bet comparison
    print("\n--- Efficiency (Profit per Bet) ---")
    for r in results:
        ppb = r['TotalProfit'] / r['Bets'] if r['Bets'] > 0 else 0
        print(f"{r['Strategy']:<20} ${ppb:.2f}/bet")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
