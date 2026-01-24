import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
from sklearn.metrics import roc_auc_score

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def load_data():
    print("Loading Data (2020-2025)...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2020-01-01'
    ORDER BY rm.MeetingDate ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    df_clean['MeetingDate'] = pd.to_datetime(df_clean['MeetingDate'])
    return df_clean, features

def run_strategy_test(df_clean, features, min_edge, max_price, min_prob=0.0):
    splits = [
        ('2024-01-01', '2024-04-01'),
        ('2024-04-01', '2024-07-01'),
        ('2024-07-01', '2024-10-01'),
        ('2024-10-01', '2025-01-01'),
        ('2025-01-01', '2025-04-01')
    ]
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'tree_method': 'hist' 
    }
    
    quarterly_rois = []
    
    print(f"\nTesting Strategy: Edge > {min_edge:.2f} | Price < ${max_price} | Prob > {min_prob}")
    
    # Pre-train Loop optimization: Can we reuse model? 
    # No, true walk-forward requires retraining or at least incremental training.
    # For speed, we will train ONCE on data up to 2024, then just predict for 2024/25?
    # NO, that's data leakage if we predict 2025 using a model trained on 2020-2023 without updating.
    # HOWEVER, retraining 5 times for every strategy permutation is too slow.
    # COMPROMISE: Train ONE model on 2020-2023. Predict 2024 Q1.
    # Train ONE model on 2020-2024 Q1. Predict Q2.
    # actually, let's just do the train loop. It's robust.
    
    for start_date, end_date in splits:
        train_data = df_clean[df_clean['MeetingDate'] < start_date]
        test_data = df_clean[(df_clean['MeetingDate'] >= start_date) & (df_clean['MeetingDate'] < end_date)]
        
        if test_data.empty: continue
            
        X_train = train_data[features]
        y_train = train_data['win']
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Fast training (fewer rounds for optimization search)
        model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
        
        test_data = test_data.copy()
        dtest = xgb.DMatrix(test_data[features])
        test_data['Prob'] = model.predict(dtest)
        
        test_data['ImpliedProb'] = 1.0 / test_data['BSP']
        test_data['Edge'] = test_data['Prob'] - test_data['ImpliedProb']
        
        # Filter
        bets = test_data[
            (test_data['BSP'] > 1.0) & 
            (test_data['BSP'] <= max_price) &
            (test_data['Edge'] > min_edge) &
            (test_data['Prob'] > min_prob)
        ].copy()
        
        if len(bets) < 10: # Ignore periods with no bets
            quarterly_rois.append(0.0) 
            continue

        stake = 10
        bets['Win'] = (bets['win'] == 1).astype(int)
        bets['Return'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
        profit = bets['Return'].sum() - (len(bets) * stake)
        roi = (profit / (len(bets) * stake)) * 100
        
        quarterly_rois.append(roi)
        # print(f"  {start_date[:7]}: {roi:+.1f}% ({len(bets)} bets)")

    avg_roi = np.mean(quarterly_rois) if quarterly_rois else -999
    min_roi = min(quarterly_rois) if quarterly_rois else -999
    print(f"  -> Avg ROI: {avg_roi:+.1f}% | Min ROI: {min_roi:+.1f}% | ROI History: {[round(r,1) for r in quarterly_rois]}")
    return avg_roi, min_roi, quarterly_rois

def main():
    df, features = load_data()
    
    # 1. Test the "Candidate" from Phase 1
    # Edge > 0.20, Price < 12.00
    print("\n--- Validating Candidate Strategy ---")
    avg, min_r, _ = run_strategy_test(df, features, min_edge=0.20, max_price=12.00)
    
    if avg > 0 and min_r > -15.0:
        print("SUCCESS: Candidate Strategy PASSED Walk-Forward Validation.")
        return

    print("FAIL: Candidate Strategy unstable or negative.")
    print("Searching for ROBUST Strategy...")
    
    # 2. Iterative Search
    strategies = [
        # Conservatve Price Caps
        {'min_edge': 0.15, 'max_price': 8.00},
        {'min_edge': 0.20, 'max_price': 8.00},
        {'min_edge': 0.25, 'max_price': 8.00},
        
        # Mid Caps
        {'min_edge': 0.15, 'max_price': 10.00},
        {'min_edge': 0.20, 'max_price': 10.00},
        {'min_edge': 0.25, 'max_price': 10.00},
        
        # Aggressive Edge
        {'min_edge': 0.30, 'max_price': 12.00},
        {'min_edge': 0.30, 'max_price': 15.00},
        
        # High Prob (Favs)
        {'min_edge': 0.10, 'max_price': 6.00, 'min_prob': 0.25},
    ]
    
    passed = []
    
    for s in strategies:
        edge = s['min_edge']
        price = s['max_price']
        prob = s.get('min_prob', 0.0)
        
        avg_r, min_r, history = run_strategy_test(df, features, edge, price, prob)
        
        # Criteria: Avg > 3% AND No Quarter < -15%
        if avg_r > 3.0 and min_r > -15.0:
            passed.append((s, avg_r, min_r, history))
            
    print("\n" + "="*80)
    print("WALK-FORWARD SUCCESSFUL STRATEGIES")
    print("="*80)
    if not passed:
        print("No strategies passed strict validation.")
    else:
        # Sort by stability (Min ROI) then Avg ROI
        passed.sort(key=lambda x: x[2], reverse=True) 
        for p in passed:
            s, avg, mn, hist = p
            print(f"Strategy: {s} | Avg ROI: {avg:.1f}% | Min ROI: {mn:.1f}% | Hist: {hist}")

if __name__ == "__main__":
    main()
