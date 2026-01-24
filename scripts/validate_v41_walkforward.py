import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
from sklearn.metrics import roc_auc_score, log_loss

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def walk_forward_validation():
    print("Loading Data (2020-2025) for Walk-Forward Validation...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, t.TrackName, rm.MeetingDate,
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
    
    # Feature Engineering
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    # Strict Cleanup
    df_clean = df.dropna(subset=features).copy()
    df_clean['MeetingDate'] = pd.to_datetime(df_clean['MeetingDate'])
    
    # Define Windows (Quarterly in 2024 onwards)
    splits = [
        ('2024-01-01', '2024-04-01'), # Q1 2024
        ('2024-04-01', '2024-07-01'), # Q2 2024
        ('2024-07-01', '2024-10-01'), # Q3 2024
        ('2024-10-01', '2025-01-01'), # Q4 2024
        ('2025-01-01', '2025-04-01')  # Q1 2025 (Current)
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION (V41) - ODDS CAP <= $15.00")
    print("Objective: Check for Overfitting & ROI Stability (Low Price Volatility)")
    print("Strategy: Train on all history BEFORE window, Test ON window.")
    print("Commission: 8% on Wins")
    print("Bet: Flat Stake $10 when Edge > 0.20 AND BSP <= 15.0")
    print("="*80)
    print(f"{'Test Period':<20} | {'Train Rows':<10} | {'AUC (Train)':<11} | {'AUC (Test)':<10} | {'Bets':<5} | {'ROI':<6}")
    print("-" * 80)
    
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
    
    for start_date, end_date in splits:
        # Train: Start of History -> Start of Window
        train_mask = df_clean['MeetingDate'] < start_date
        # Test: Start of Window -> End of Window
        test_mask = (df_clean['MeetingDate'] >= start_date) & (df_clean['MeetingDate'] < end_date)
        
        train_data = df_clean[train_mask]
        test_data = df_clean[test_mask]
        
        if test_data.empty:
            print(f"{start_date} -> {end_date}: No Test Data")
            continue
            
        X_train = train_data[features]
        y_train = train_data['win']
        X_test = test_data[features]
        y_test = test_data['win']
        
        # Train
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(params, dtrain, num_boost_round=600, verbose_eval=False)
        
        # Evaluate AUC
        train_preds = model.predict(dtrain)
        test_preds = model.predict(dtest)
        
        auc_train = roc_auc_score(y_train, train_preds)
        auc_test = roc_auc_score(y_test, test_preds)
        
        # Evaluate Profitability (Edge > 0.20 & Odds Cap)
        test_data = test_data.copy()
        test_data['Prob'] = test_preds
        test_data['ImpliedProb'] = 1.0 / test_data['BSP']
        test_data['Edge'] = test_data['Prob'] - test_data['ImpliedProb']
        
        # Filter: Valid BSP & High Edge & ODDS CAP
        bets = test_data[(test_data['BSP'] > 1.0) & 
                         (test_data['BSP'] <= 15.0) &
                         (test_data['Edge'] > 0.20)].copy()
        
        roi_str = "0.0%"
        bet_count = len(bets)
        
        if bet_count > 0:
            stake = 10
            # 8% Commission
            bets['Win'] = (bets['win'] == 1).astype(int)
            bets['Return'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
            profit = bets['Return'].sum() - (bet_count * stake)
            roi = (profit / (bet_count * stake)) * 100
            
            roi_str = f"{roi:+.1f}%"
        
        print(f"{start_date[:7]} to {end_date[:7]:<7} | {len(X_train):<10} | {auc_train:.4f}      | {auc_test:.4f}     | {bet_count:<5} | {roi_str}")
        
        results.append({
            'Period': f"{start_date[:7]}",
            'ROI': roi if bet_count > 0 else 0
        })

    print("-" * 80)
    
    # ROI Stability
    rois = [r['ROI'] for r in results]
    print(f"\nROI STABILITY (Quarterly Capped): {rois}")

if __name__ == "__main__":
    walk_forward_validation()
