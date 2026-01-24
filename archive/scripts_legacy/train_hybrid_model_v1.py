"""
Train Hybrid XGBoost Model (Market + Trainer + Pace)
Goal: Predict Win Probability and find Value Bets (Model > Market).
Training: 2023-2024
Testing: 2025
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading Data (2023-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        ge.TrainerID,
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
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Features...")
    
    # 1. BENCHMARKS
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    
    # 2. DOG HISTORY (Rolling)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    df['DogPaceAvg'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df['DogSplitAvg'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    df['DaysSince'] = (df['MeetingDate'] - g['MeetingDate'].shift(1)).dt.days.fillna(999).clip(upper=60)
    
    # 3. TRAINER STATS (Expanding Window)
    print("  Calculating Trainer Stats (Expanding)...")
    df = df.sort_values(['TrainerID', 'MeetingDate'])
    df['TrainerWinRate'] = df.groupby('TrainerID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0) # Default to 0 for new trainers? Or global avg? 0 is safe conservative.
    
    # 4. MARKET
    df['ImpliedProb'] = 1 / df['Odds']
    
    # Filter for valid rows (Must have Odds, Hist, etc)
    df = df.dropna(subset=['Odds', 'DogPaceAvg', 'ImpliedProb']).copy()
    
    return df

def train_and_validate(df):
    print("\nTraining Model...")
    
    # Split
    train_mask = (df['MeetingDate'].dt.year >= 2023) & (df['MeetingDate'].dt.year <= 2024)
    test_mask = (df['MeetingDate'].dt.year == 2025)
    
    train_df = df[train_mask]
    test_df = df[test_mask].copy()
    
    features = ['ImpliedProb', 'TrainerWinRate', 'DogPaceAvg', 'DogSplitAvg', 'Box', 'DaysSince']
    target = 'IsWin'
    
    print(f"  Train Size: {len(train_df)}")
    print(f"  Test Size:  {len(test_df)}")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(train_df[features], train_df[target])
    
    # Feature Importance
    print("\nFeature Importance:")
    for name, imp in zip(features, model.feature_importances_):
        print(f"  {name}: {imp:.4f}")
        
    # Predict
    probs = model.predict_proba(test_df[features])[:, 1]
    test_df['ModelProb'] = probs
    
    # Metrics
    auc = roc_auc_score(test_df[target], probs)
    ll = log_loss(test_df[target], probs)
    print(f"\nTest Metrics (2025): AUC={auc:.4f}, LogLoss={ll:.4f}")
    
    return test_df

def evaluate_strategy(df):
    print("\n" + "="*80)
    print("VALUE BETTING STRATEGY (2025)")
    print("="*80)
    
    # Strategy: Bet if ModelProb > ImpliedProb * Threshold
    # We want +EV bets.
    
    thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    print(f"{'Threshold':<10} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'Profit':<10} {'ROI%':<10}")
    print("-" * 70)
    
    for thresh in thresholds:
        # Edge calculation
        # If Model says 40% and Market says 30% (Implied 0.33), Edge = 40/33 = 1.21
        mask = (df['ModelProb'] > df['ImpliedProb'] * thresh) & (df['Odds'] >= 1.50) & (df['Odds'] <= 30)
        bets = df[mask].copy()
        
        if len(bets) == 0:
            continue
            
        n_bets = len(bets)
        wins = bets['IsWin'].sum()
        strike = wins / n_bets * 100
        
        # Calculate Profit
        # Winners: (Odds - 1) * 1 unit
        # Losers: -1 unit
        winner_profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum()
        loser_loss = (n_bets - wins)
        profit = winner_profit - loser_loss
        
        roi = profit / n_bets * 100
        
        print(f"{thresh:<10} {n_bets:<8} {wins:<8} {strike:<10.1f} {profit:<10.1f} {roi:<10.1f}")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    test_df = train_and_validate(df)
    evaluate_strategy(test_df)
