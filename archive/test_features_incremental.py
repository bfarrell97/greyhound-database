"""
Test features one at a time to find improvements
Start with baseline ($1.50-$2.00 range, +1.29% ROI baseline)
Test:
1. BoxDraw feature alone
2. WeightClass feature alone
3. Both together
4. Gentle recency [2.5, 1.5, 1.0, 0.75, 0.5] instead of [2.0, 1.5, 1.0, 1.0, 1.0]
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import xgboost as xgb
from sklearn.model_selection import train_test_split

DB_PATH = 'greyhound_racing.db'
TRAIN_START = '2023-01-01'
TRAIN_END = '2025-05-31'
TEST_START = '2025-01-01'
TEST_END = '2025-11-30'
CONFIDENCE_THRESHOLD = 0.80
INITIAL_BANKROLL = 1000.0

print("="*80)
print("SYSTEMATIC FEATURE TESTING: One feature at a time")
print("="*80)

# Test different feature configurations
test_configs = [
    {
        'name': 'BASELINE (8 features)',
        'features': ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'],
        'recency': [2.0, 1.5, 1.0, 1.0, 1.0],
        'use_boxdraw': False,
        'use_weightclass': False,
    },
    {
        'name': 'BOXDRAW FEATURE ONLY',
        'features': ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 'BoxDraw', 'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'],
        'recency': [2.0, 1.5, 1.0, 1.0, 1.0],
        'use_boxdraw': True,
        'use_weightclass': False,
    },
    {
        'name': 'WEIGHTCLASS FEATURE ONLY',
        'features': ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 'WeightClass', 'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'],
        'recency': [2.0, 1.5, 1.0, 1.0, 1.0],
        'use_boxdraw': False,
        'use_weightclass': True,
    },
    {
        'name': 'BOTH FEATURES (BoxDraw + WeightClass)',
        'features': ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 'BoxDraw', 'WeightClass', 'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'],
        'recency': [2.0, 1.5, 1.0, 1.0, 1.0],
        'use_boxdraw': True,
        'use_weightclass': True,
    },
    {
        'name': 'GENTLE RECENCY [2.5, 1.5, 1.0, 0.75, 0.5]',
        'features': ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'],
        'recency': [2.5, 1.5, 1.0, 0.75, 0.5],
        'use_boxdraw': False,
        'use_weightclass': False,
    },
]

results = []

# Connect to database
conn = sqlite3.connect(DB_PATH)
ml_model = GreyhoundMLModel()

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")
    
    # Load training data
    print("Loading training data...")
    query = """
    SELECT DISTINCT
        ge.EntryID,
        g.GreyhoundName,
        g.GreyhoundID,
        rm.MeetingDate,
        t.TrackName as CurrentTrack,
        t.TrackID,
        r.RaceNumber,
        r.Distance,
        ge.Box,
        ge.Weight,
        ge.Position,
        ge.StartingPrice,
        ge.SplitBenchmarkLengths as G_Split_ADJ,
        rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
        ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
        rm.MeetingAvgBenchmarkLengths as M_OT_ADJ
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= ? AND rm.MeetingDate <= ?
      AND ge.Position IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
      AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT LIKE '%TAS%'
    ORDER BY rm.MeetingDate, r.RaceNumber, ge.Box
    """
    
    df = pd.read_sql_query(query, conn, params=(TRAIN_START, TRAIN_END))
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['IsWinner'] = (df['Position'] == 1).astype(int)
    
    # Load historical data for feature extraction
    hist_query = """
        SELECT
            ge.GreyhoundID,
            t.TrackName,
            r.Distance,
            ge.Weight,
            ge.Position,
            ge.FinishTime,
            ge.SplitBenchmarkLengths as G_Split_ADJ,
            rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
            ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
            rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
            rm.MeetingDate
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
          AND ge.SplitBenchmarkLengths IS NOT NULL
          AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT LIKE '%TAS%'
        ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
    """
    
    hist_df = pd.read_sql_query(hist_query, conn, params=(TRAIN_END,))
    
    # Calculate box win rates
    box_stats_query = """
        SELECT
            t.TrackID,
            r.Distance,
            ge.Box,
            COUNT(*) as TotalRaces,
            SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
          AND ge.Box IS NOT NULL
          AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT LIKE '%TAS%'
        GROUP BY t.TrackID, r.Distance, ge.Box
    """
    box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(TRAIN_END,))
    box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']
    box_win_rates = {}
    for _, row in box_stats_df.iterrows():
        key = (row['TrackID'], row['Distance'], row['Box'])
        box_win_rates[key] = row['BoxWinRate']
    
    # Extract features with configuration
    print(f"  Extracting features with recency {config['recency']}...")
    hist_grouped = hist_df.groupby('GreyhoundID')
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 100000 == 0 and idx > 0:
            print(f"    Processed {idx:,}/{len(df):,}...")
        
        if row['GreyhoundID'] not in hist_grouped.groups:
            continue
        
        greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
        greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
        last_5 = greyhound_hist.head(5)
        
        if len(last_5) < 5:
            continue
        
        features = {}
        features['EntryID'] = row['EntryID']
        features['IsWinner'] = row['IsWinner']
        features['StartingPrice'] = row['StartingPrice']
        
        # BoxWinRate
        box_key = (row['TrackID'], row['Distance'], row['Box'])
        features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)
        
        # Recent form
        last_3 = last_5.head(3)
        if len(last_3) > 0:
            last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
            features['AvgPositionLast3'] = last_3_positions.mean()
            features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
        else:
            features['AvgPositionLast3'] = 4.5
            features['WinRateLast3'] = 0
        
        # Optional features
        if config['use_boxdraw']:
            features['BoxDraw'] = row['Box'] / 8.0
        
        if config['use_weightclass']:
            weight = row['Weight']
            if pd.notna(weight):
                if weight < 30:
                    features['WeightClass'] = 0
                elif weight <= 33:
                    features['WeightClass'] = 1
                else:
                    features['WeightClass'] = 2
            else:
                features['WeightClass'] = 1
        
        # GM_OT_ADJ with specified recency
        recency_weights = config['recency']
        for i, (_, race) in enumerate(last_5.iterrows(), 1):
            gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
            track_weight = ml_model.get_track_tier_weight(race['TrackName'])
            recency_weight = recency_weights[i - 1]
            features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight * recency_weight
        
        features_list.append(features)
    
    train_df = pd.DataFrame(features_list)
    print(f"  Extracted {len(train_df):,} training examples")
    
    # Train model
    print("  Training XGBoost model...")
    X = train_df[config['features']]
    y = train_df['IsWinner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        tree_method='hist',
        device='cpu',
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    model.fit(X_train, y_train)
    
    # Load test data (2025)
    print("  Loading test data...")
    query_test = """
    SELECT DISTINCT
        ge.EntryID,
        g.GreyhoundName,
        g.GreyhoundID,
        rm.MeetingDate,
        t.TrackName as CurrentTrack,
        t.TrackID,
        r.RaceNumber,
        r.Distance,
        ge.Box,
        ge.Weight,
        ge.Position,
        ge.StartingPrice,
        ge.SplitBenchmarkLengths as G_Split_ADJ,
        rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
        ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
        rm.MeetingAvgBenchmarkLengths as M_OT_ADJ
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= ? AND rm.MeetingDate <= ?
      AND ge.Position IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
      AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
      AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT LIKE '%TAS%'
    ORDER BY rm.MeetingDate, r.RaceNumber, ge.Box
    """
    
    test_raw = pd.read_sql_query(query_test, conn, params=(TEST_START, TEST_END))
    test_raw['Position'] = pd.to_numeric(test_raw['Position'], errors='coerce')
    test_raw['IsWinner'] = (test_raw['Position'] == 1).astype(int)
    
    # Extract test features with same configuration
    print("  Extracting test features...")
    hist_query_test = """
        SELECT
            ge.GreyhoundID,
            t.TrackName,
            r.Distance,
            ge.Weight,
            ge.Position,
            ge.FinishTime,
            ge.SplitBenchmarkLengths as G_Split_ADJ,
            rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
            ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
            rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
            rm.MeetingDate
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
          AND ge.SplitBenchmarkLengths IS NOT NULL
          AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND t.TrackName NOT IN ('Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North', 'Launceston', 'Hobart', 'Devonport')
          AND t.TrackName NOT LIKE '%NZ%'
          AND t.TrackName NOT LIKE '%TAS%'
        ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
    """
    
    hist_test = pd.read_sql_query(hist_query_test, conn, params=(TEST_START,))
    hist_test_grouped = hist_test.groupby('GreyhoundID')
    
    test_features_list = []
    for idx, row in test_raw.iterrows():
        if row['GreyhoundID'] not in hist_test_grouped.groups:
            continue
        
        greyhound_hist = hist_test_grouped.get_group(row['GreyhoundID'])
        greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
        last_5 = greyhound_hist.head(5)
        
        if len(last_5) < 5:
            continue
        
        features = {}
        features['EntryID'] = row['EntryID']
        features['IsWinner'] = row['IsWinner']
        features['StartingPrice'] = row['StartingPrice']
        
        box_key = (row['TrackID'], row['Distance'], row['Box'])
        features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)
        
        last_3 = last_5.head(3)
        if len(last_3) > 0:
            last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
            features['AvgPositionLast3'] = last_3_positions.mean()
            features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
        else:
            features['AvgPositionLast3'] = 4.5
            features['WinRateLast3'] = 0
        
        if config['use_boxdraw']:
            features['BoxDraw'] = row['Box'] / 8.0
        
        if config['use_weightclass']:
            weight = row['Weight']
            if pd.notna(weight):
                if weight < 30:
                    features['WeightClass'] = 0
                elif weight <= 33:
                    features['WeightClass'] = 1
                else:
                    features['WeightClass'] = 2
            else:
                features['WeightClass'] = 1
        
        recency_weights = config['recency']
        for i, (_, race) in enumerate(last_5.iterrows(), 1):
            gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
            track_weight = ml_model.get_track_tier_weight(race['TrackName'])
            recency_weight = recency_weights[i - 1]
            features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight * recency_weight
        
        test_features_list.append(features)
    
    test_df = pd.DataFrame(test_features_list)
    print(f"  Extracted {len(test_df):,} test examples")
    
    # Make predictions on test set, filter $1.50-$2.00
    X_test_data = test_df[config['features']]
    predictions = model.predict_proba(X_test_data)[:, 1]
    test_df['Probability'] = predictions
    test_df['ImpliedProb'] = 1 / test_df['StartingPrice']
    test_df['Value'] = test_df['Probability'] > test_df['ImpliedProb']
    
    # Filter for high confidence value bets in $1.50-$2.00
    high_conf = test_df[test_df['Probability'] >= CONFIDENCE_THRESHOLD]
    value_bets = high_conf[high_conf['Value']]
    bracket = value_bets[(value_bets['StartingPrice'] >= 1.50) & (value_bets['StartingPrice'] < 2.00)]
    
    if len(bracket) > 0:
        # Calculate ROI
        stake_pct = 0.02
        total_staked = len(bracket) * INITIAL_BANKROLL * stake_pct
        bracket_copy = bracket.copy()
        bracket_copy['Stake'] = INITIAL_BANKROLL * stake_pct
        bracket_copy['Return'] = np.where(bracket_copy['IsWinner'] == 1, bracket_copy['StartingPrice'] * bracket_copy['Stake'], 0)
        bracket_copy['PnL'] = bracket_copy['Return'] - bracket_copy['Stake']
        
        wins = bracket_copy['IsWinner'].sum()
        total_pnl = bracket_copy['PnL'].sum()
        roi = (total_pnl / total_staked) * 100
        strike = (wins / len(bracket_copy)) * 100
        
        result = {
            'Config': config['name'],
            'Bets': len(bracket_copy),
            'Wins': wins,
            'Strike%': strike,
            'ROI%': roi,
            'P/L': total_pnl,
        }
        results.append(result)
        
        print(f"  RESULTS ($1.50-$2.00):")
        print(f"    Bets: {len(bracket_copy)}, Wins: {wins}, Strike: {strike:.1f}%")
        print(f"    P/L: ${total_pnl:.2f}, ROI: {roi:.2f}%")
    else:
        print(f"  No bets found in $1.50-$2.00 range")

conn.close()

# Summary
print("\n" + "="*80)
print("SUMMARY: Feature Testing Results ($1.50-$2.00 range)")
print("="*80)
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROI%', ascending=False)
    print(f"\n{'Config':<40} {'Bets':>6} {'Wins':>5} {'Strike%':>8} {'ROI%':>8}")
    print("-"*80)
    for _, row in results_df.iterrows():
        print(f"{row['Config']:<40} {row['Bets']:>6.0f} {row['Wins']:>5.0f} {row['Strike%']:>7.1f}% {row['ROI%']:>7.2f}%")
    
    best = results_df.iloc[0]
    print(f"\n✅ BEST CONFIGURATION: {best['Config']}")
    print(f"   ROI: {best['ROI%']:.2f}% (vs baseline {results_df.iloc[-1]['ROI%']:.2f}%)")
else:
    print("No results to display")
