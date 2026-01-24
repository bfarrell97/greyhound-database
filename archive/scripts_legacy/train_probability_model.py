"""
Train Win Probability Model (XGBClassifier)
Objective: Predict Probability of Winning (0-1) to identify value bets (Edge > 0).
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle

DB_PATH = 'greyhound_racing.db'

def progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def train_prob_model():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. LOAD DATA
    progress("Loading race data...")
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split,
        ge.FinishTime,
        ge.Position
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND ge.Split IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Target'] = (df['Position'] == '1').astype(int)
    
    progress(f"Loaded {len(df):,} records")
    
    # 2. FEATURE ENGINEERING (Reusing successful normalization logic)
    progress("Calculating features...")
    
    # Calculate Track/Dist Medians (Benchmarks)
    # Using global medians for simplicity and robustness
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    # Sort for rolling ops
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # --- Features ---
    # We want to know a dog's "Recent Form" entering the race.
    # We use shift(1) so we are using PREVIOUS races to predict CURRENT result.
    g = df.groupby('GreyhoundID')
    
    # SPLIT Stats
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # PACE Stats
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    # Days Since
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999)
    # Cap days since at 60 to prevent outliers skewing
    df['DaysSince'] = df['DaysSince'].clip(upper=60)
    
    # Drop rows without history
    model_df = df.dropna(subset=['s_Roll5', 'p_Roll5']).copy()
    
    progress(f"Training set size: {len(model_df):,} records")
    
    # 3. SPLIT TRAIN/TEST
    train_mask = model_df['MeetingDate'] < '2024-01-01'
    test_mask = model_df['MeetingDate'] >= '2024-01-01'
    
    features = ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    target = 'Target'
    
    X_train = model_df.loc[train_mask, features]
    y_train = model_df.loc[train_mask, target]
    
    X_test = model_df.loc[test_mask, features]
    y_test = model_df.loc[test_mask, target]
    
    # 4. TRAIN XGBOOST CLASSIFIER
    progress("Training XGBClassifier...")
    # REMOVED scale_pos_weight to ensure calibrated probabilities
    # We want natural probability distribution (e.g. 0.125 avg), not balanced 0.5
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight=ratio,  <-- Caused over-confidence
        eval_metric='logloss',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 5. EVALUATE
    progress("Evaluating performance...")
    
    probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probs)
    logloss = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    
    print("\n" + "="*60)
    print("WIN PROBABILITY MODEL RESULTS (2024-2025 Test Set)")
    print("="*60)
    print(f"ROC AUC:      {roc_auc:.4f} (0.5=Random, 1.0=Perfect)")
    print(f"Log Loss:     {logloss:.4f} (Lower is better)")
    print(f"Brier Score:  {brier:.4f} (Lower is better)")
    
    # Calibration Check
    print("\nCalibration (Predicted vs Actual Win %):")
    cal_df = pd.DataFrame({'prob': probs, 'actual': y_test})
    cal_df['bin'] = pd.qcut(cal_df['prob'], 10)
    calibration = cal_df.groupby('bin')['actual'].mean()
    print(calibration)
    
    print("\n--- Feature Importance ---")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))
    
    # Save Model
    with open('models/prob_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to models/prob_xgb_model.pkl")

if __name__ == "__main__":
    train_prob_model()
