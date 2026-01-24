"""
Train Race Shape Model (Early, Late, Overall Speed) (2025)
Goal: Predict Full Race Shape to identify Dominant Leaders and Swoopers.
Models: 3 Separate XGBoost Regressors.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading Data (2023-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.SireID,
        r.RaceID,
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
    WHERE rm.MeetingDate >= '2023-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    # Run Home
    df['RunHome'] = df['FinishTime'] - df['Split']
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Features (Race Shape)...")
    
    # 1. Benchmarks (Global)
    cols = ['FinishTime', 'Split', 'RunHome']
    for c in cols:
        bench_col = f'TrackDistMedian{c}'
        bench = df.groupby(['TrackName', 'Distance'])[c].median().reset_index()
        bench.columns = ['TrackName', 'Distance', bench_col]
        df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
        df[f'Norm{c}'] = df[c] - df[bench_col]
        
    # 2. Rolling Avgs (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # Features for each component
    for c in ['NormFinishTime', 'NormSplit', 'NormRunHome']:
        df[f'Dog{c}Avg'] = g[c].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
    # 3. Sire Stats (General Win Rate) - Expanding
    print("  Calculating Sire Stats...")
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    # Filter for valid rows (Must have Split for valid Split/RunHome training)
    # But for Overall, Split might not be needed? 
    # Let's demand Split exists to have a complete shape.
    df = df.dropna(subset=['Split', 'FinishTime', 'RunHome']).copy()
    
    return df

def train_and_predict(df):
    
    train_mask = (df['MeetingDate'].dt.year >= 2023) & (df['MeetingDate'].dt.year <= 2024)
    test_mask = (df['MeetingDate'].dt.year == 2025)
    
    train_df = df[train_mask]
    test_df = df[test_mask].copy()
    
    print(f"\nTraining Size: {len(train_df)}")
    print(f"Test Size:     {len(test_df)}")
    
    # Common Features
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'SireWinRate', 'Box', 'Distance']
    
    targets = {
        'Split': 'NormSplit',
        'RunHome': 'NormRunHome',
        'Overall': 'NormFinishTime'
    }
    
    models = {}
    
    for name, target in targets.items():
        print(f"  Training {name} Model...")
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        
        # Filter train for target existence (should be handled by dropna above, but safe check)
        t_df = train_df.dropna(subset=[target])
        model.fit(t_df[features], t_df[target])
        models[name] = model
        
        # Predict
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    return test_df

def test_strategies(df):
    print("\n" + "="*80)
    print("RACE SHAPE STRATEGY PERFORMANCE (2025)")
    print("="*80)
    
    # Define RaceKey
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Filter Field Size
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6].copy()
    
    # Calculate Ranks from Predictions
    # Lower is Better (Faster time, Faster split)
    df['PredSplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min')
    df['PredRunHomeRank'] = df.groupby('RaceKey')['PredRunHome'].rank(method='min')
    df['PredOverallRank'] = df.groupby('RaceKey')['PredOverall'].rank(method='min')
    
    strategies = {
        # 1. THE RABBIT: True Leader (Split 1)
        'Rabbit (Pred Split 1)': (df['PredSplitRank'] == 1),
        
        # 2. THE DOMINATOR: Split 1 AND Overall 1
        'Dominator (Split 1 + Overall 1)': (df['PredSplitRank'] == 1) & (df['PredOverallRank'] == 1),
        
        # 3. THE SWOOPER: RunHome 1 AND Overall 1 (But NOT Split 1)
        'Swooper (RunHome 1 + Overall 1)': (df['PredRunHomeRank'] == 1) & (df['PredOverallRank'] == 1) & (df['PredSplitRank'] > 1),
        
        # 4. STEAMROLLER: RunHome 1 (Pure Late Speed)
        'Steamroller (Pred RunHome 1)': (df['PredRunHomeRank'] == 1),
        
        # 5. PURE CLASS: Overall 1
        'Class (Pred Overall 1)': (df['PredOverallRank'] == 1),
    }
    
    print(f"{'Strategy':<40} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'Profit':<10} {'ROI%':<10}")
    print("-" * 88)
    
    for name, mask in strategies.items():
        # Apply Odds Filter
        bet_mask = mask & (df['Odds'] >= 2.0) & (df['Odds'] <= 30)
        bets = df[bet_mask].copy()
        
        if len(bets) == 0:
            print(f"{name:<40} 0 bets")
            continue
            
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        strike = wins / n_bets * 100
        
        winner_profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum()
        loser_loss = (n_bets - wins)
        profit = winner_profit - loser_loss
        roi = profit / n_bets * 100
        
        print(f"{name:<40} {n_bets:<8} {wins:<8} {strike:<10.1f} {profit:<10.1f} {roi:<10.1f}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    test_df = train_and_predict(df)
    test_strategies(test_df)
