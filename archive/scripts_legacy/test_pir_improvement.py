"""
EXPERIMENT: PIR Heuristic vs XGBoost
Can an ML model predict the split leader better than the simple (Avg + BoxAdj) formula?
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

DB_PATH = 'greyhound_racing.db'

def run_experiment():
    print("="*80)
    print("PIR EXPERIMENT: HEURISTIC vs XGBOOST")
    print("="*80)

    conn = sqlite3.connect(DB_PATH)

    # 1. Load Data (Historical Splits)
    print("Loading data...")
    query = """
    SELECT 
        ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
        r.Distance, t.TrackID, rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.FirstSplitPosition IS NOT NULL
      AND ge.Box IS NOT NULL
      AND rm.MeetingDate >= '2022-01-01'
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert to numeric
    df['FirstSplitPosition'] = pd.to_numeric(df['FirstSplitPosition'], errors='coerce')
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    df = df.dropna(subset=['FirstSplitPosition', 'Box'])
    
    print(f"Loaded {len(df):,} rows")

    # 2. Feature Engineering (Historical Avg Split)
    print("Calculating historical features...")
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Rolling Avg Split (Last 5) - SHIFTED so no leakage
    df['HistAvgSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    
    # Filter for valid history
    df_valid = df.dropna(subset=['HistAvgSplit']).copy()
    print(f"Rows with history: {len(df_valid):,}")

    # 3. Label Engineering (Is Actual Leader?)
    # We need to know who ACTUALLY led the race (SplitPos == 1)
    df_valid['IsActualLeader'] = (df_valid['FirstSplitPosition'] == 1).astype(int)

    # 4. BASELINE: Heuristic Prediction
    # Formula: AvgSplit + BoxAdj
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df_valid['BoxAdj'] = df_valid['Box'].map(box_adj).fillna(0)
    df_valid['HeuristicScore'] = df_valid['HistAvgSplit'] + df_valid['BoxAdj']
    
    # 5. ML PREDICTION: XGBoost
    # Features: HistAvgSplit, Box, Distance
    features = ['HistAvgSplit', 'Box', 'Distance']
    X = df_valid[features]
    y = df_valid['IsActualLeader']

    # Time-based Split (Train < 2024, Test >= 2024)
    # This simulates real-world usage better than random split
    train_mask = df_valid['MeetingDate'] < '2024-01-01'
    test_mask = df_valid['MeetingDate'] >= '2024-01-01'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTraining XGBoost on {len(X_train):,} rows (2022-2023)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Testing on {len(X_test):,} rows (2024+)...")
    
    # Get ML Probabilities
    # We want "Probability of Leading", low score is better for heuristic, high prob is better for ML
    # To compare, we rank within each race.
    df_test = df_valid[test_mask].copy()
    df_test['ML_Prob'] = model.predict_proba(X_test)[:, 1]
    
    # 6. COMPARISON: Rank within Race
    print("\nComparing predictions race-by-race...")
    
    # We need to group by RaceID to see who is predicted Rank 1
    # Only consider races with at least 6 dogs for fair comparison
    race_counts = df_test['RaceID'].value_counts()
    valid_races = race_counts[race_counts >= 6].index
    
    df_eval = df_test[df_test['RaceID'].isin(valid_races)].copy()
    
    # Rank Heuristic (Lowest Score = Rank 1)
    df_eval['HeuristicRank'] = df_eval.groupby('RaceID')['HeuristicScore'].rank(method='min', ascending=True)
    
    # Rank ML (Highest Prob = Rank 1)
    df_eval['MLRank'] = df_eval.groupby('RaceID')['ML_Prob'].rank(method='min', ascending=False)
    
    # Check if predicted leader was the ACTUAL leader
    # Note: Actual Leader is whoever had FirstSplitPosition == 1
    # Sometimes multiple dogs share pos 1? Usually not.
    # We define success: Did the Rank 1 prediction match a dog with Actual SplitPos=1?
    
    # Filter to Predicted Winners
    heuristic_winners = df_eval[df_eval['HeuristicRank'] == 1]
    ml_winners = df_eval[df_eval['MLRank'] == 1]
    
    # Calculate Accuracy
    h_acc = heuristic_winners['IsActualLeader'].mean() * 100
    m_acc = ml_winners['IsActualLeader'].mean() * 100
    
    print(f"\nAnalyzed {len(valid_races):,} races")
    print("-" * 40)
    print(f"Heuristic Accuracy:  {h_acc:.2f}%")
    print(f"XGBoost Accuracy:    {m_acc:.2f}%")
    print("-" * 40)
    
    diff = m_acc - h_acc
    if diff > 1.0:
        print(f"RESULT: XGBoost wins significantly (+{diff:.2f}%)")
    elif diff < -1.0:
        print(f"RESULT: Heuristic wins significantly ({diff:.2f}%)")
    else:
        print(f"RESULT: Tie / Negligible difference ({diff:+.2f}%)")
        print("Recommendation: Stick with Heuristic (Simple is better)")

    # Feature Importance
    print("\nXGBoost Feature Importance:")
    imps = dict(zip(features, model.feature_importances_))
    for k, v in sorted(imps.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    run_experiment()
