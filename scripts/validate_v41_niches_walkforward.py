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

def validate_niches():
    print("Loading Data (2020-2025) for Niche Validation...")
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
    
    # Feature Engineering
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    # Strict Cleanup
    df_clean = df.dropna(subset=features).copy()
    df_clean['MeetingDate'] = pd.to_datetime(df_clean['MeetingDate'])
    
    # Define Windows
    splits = [
        ('2024-01-01', '2024-04-01'),
        ('2024-04-01', '2024-07-01'),
        ('2024-07-01', '2024-10-01'),
        ('2024-10-01', '2025-01-01'),
        ('2025-01-01', '2025-04-01')
    ]
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: 'GOLDEN RULES' NICHE")
    print("Filters: Age<=2y, Low Grade, Circle Tracks")
    print("Strategy: Edge > 0.20 | Comm: 8%")
    print("="*80)
    print(f"{'Period':<15} | {'Bets':<6} | {'Winners':<7} | {'Strike':<6} | {'Profit':<10} | {'ROI':<6}")
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
    
    results = []

    for start_date, end_date in splits:
        # Train/Test Split
        train_data = df_clean[df_clean['MeetingDate'] < start_date]
        test_data = df_clean[(df_clean['MeetingDate'] >= start_date) & (df_clean['MeetingDate'] < end_date)]
        
        if test_data.empty: continue
            
        # Train
        X_train = train_data[features]
        y_train = train_data['win']
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(params, dtrain, num_boost_round=600, verbose_eval=False)
        
        # Predict
        test_data = test_data.copy()
        dtest = xgb.DMatrix(test_data[features])
        test_data['Prob'] = model.predict(dtest)
        
        # Apply Filters ( The Golden Rules )
        # 1. Age <= 2 years (730 days)
        test_data = test_data[test_data['DogAgeDays'] <= 730]
        
        # 2. Exclude High Grades
        def is_high_grade(g):
            g = str(g).lower()
            return '1' in g or 'free' in g or 'best' in g or 'invitation' in g
            
        test_data = test_data[~test_data['Grade'].apply(is_high_grade)]
        
        # 3. Exclude Straight Tracks
        test_data = test_data[~test_data['TrackName'].isin(['Capalaba', 'Healesville', 'Murray Bridge Straight'])]
        
        # 4. Strategy: Edge > 0.20
        test_data['ImpliedProb'] = 1.0 / test_data['BSP']
        test_data['Edge'] = test_data['Prob'] - test_data['ImpliedProb']
        
        bets = test_data[(test_data['BSP'] > 1.0) & (test_data['Edge'] > 0.20)].copy()
        
        # ROI Calc
        bet_count = len(bets)
        roi_str = "0.0%"
        profit = 0
        
        if bet_count > 0:
            stake = 10
            bets['Win'] = (bets['win'] == 1).astype(int)
            bets['Return'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92 + stake, 0)
            profit = bets['Return'].sum() - (bet_count * stake)
            roi = (profit / (bet_count * stake)) * 100
            roi_str = f"{roi:+.1f}%"
            
            results.append(roi)
            
        print(f"{start_date[:7]:<15} | {bet_count:<6} | {bets['Win'].sum():<7} | {bets['Win'].mean()*100:>5.1f}% | ${profit:>9,.2f} | {roi_str}")

    print("-" * 80)
    if results:
        print(f"Average Quarterly ROI: {np.mean(results):.2f}%")

if __name__ == "__main__":
    validate_niches()
