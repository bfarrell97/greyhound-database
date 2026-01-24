import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def train_v44_classifier():
    print("Loading Data (2020-2025) for V44 Steamer Classifier...")
    print("Split Strategy: Train (2020-2023) | Test (2024-2025)")
    
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, 
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2020-01-01' 
    AND ge.Price5Min IS NOT NULL
    AND ge.BSP > 0
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Base Features (V41)
    # Note: V41 FE usually handles scaling. We assume it handles 5 years ok.
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features_v41 = fe.get_feature_list()
    
    # Clean Initial
    df_clean = df.dropna(subset=['Price5Min', 'BSP']).copy()
    
    # 2. Define Steamer Target
    # MoveRatio > 1.15 (15% drop)
    df_clean['MoveRatio'] = df_clean['Price5Min'] / df_clean['BSP']
    df_clean['Is_Steamer'] = (df_clean['MoveRatio'] > 1.15).astype(int)
    
    # 3. ADVANCED LAG FEATURES
    print("Engineering Historical Steam Features...")
    df_clean = df_clean.sort_values('MeetingDate')
    
    # A. Dog Steam Rate
    df_clean['Prev_Steam'] = df_clean.groupby('GreyhoundID')['Is_Steamer'].shift(1)
    df_clean['Dog_Rolling_Steam_10'] = df_clean.groupby('GreyhoundID')['Prev_Steam'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    ).fillna(0)
    
    # B. Trainer Steam Rate
    df_clean['Trainer_Prev_Steam'] = df_clean.groupby('TrainerID')['Is_Steamer'].shift(1)
    df_clean['Trainer_Rolling_Steam_50'] = df_clean.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
        lambda x: x.rolling(window=50, min_periods=10).mean()
    ).fillna(0)
    
    # 4. Base Model Probabilities
    print("Generating Base Model Probabilities...")
    model_v41 = joblib.load('models/xgb_v41_final.pkl')
    
    for c in features_v41:
        if c not in df_clean.columns: df_clean[c] = 0
            
    dtest_v41 = xgb.DMatrix(df_clean[features_v41])
    df_clean['V41_Prob'] = model_v41.predict(dtest_v41)
    df_clean['V41_Price'] = 1.0 / df_clean['V41_Prob']
    
    # Discrepancy Features
    df_clean['Discrepancy'] = df_clean['Price5Min'] / df_clean['V41_Price']
    df_clean['Price_Diff'] = df_clean['Price5Min'] - df_clean['V41_Price']
    
    # 5. TRAIN / TEST SPLIT (DATE BASED)
    train_mask = df_clean['MeetingDate'] < '2024-01-01'
    test_mask = df_clean['MeetingDate'] >= '2024-01-01'
    
    features_v44 = [
        'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
        'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
    ]
    
    X_train = df_clean.loc[train_mask, features_v44]
    y_train = df_clean.loc[train_mask, 'Is_Steamer']
    
    X_test = df_clean.loc[test_mask, features_v44]
    y_test = df_clean.loc[test_mask, 'Is_Steamer']
    
    print(f"Train Set (2020-2023): {len(X_train)} rows")
    print(f"Test Set (2024-2025):  {len(X_test)} rows")
    print(f"Train Steamer Rate: {y_train.mean()*100:.2f}%")
    print(f"Test Steamer Rate:  {y_test.mean()*100:.2f}%")
    
    # 6. Train XGBoost V44
    print("Training XGBoost V44...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    
    # Importance
    imp = pd.Series(model.feature_importances_, index=features_v44).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(imp.head(10))
    
    # Threshold Analysis
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print("\n" + "="*80)
    print(f"{'Thres':<6} | {'Count':<6} | {'Precision':<10} | {'Win Rate':<10} | {'Avg Price':<10} | {'ROI (Sim)':<10}")
    print("-" * 80)
    
    best_roi = -100
    
    # Test Dataframe for Simulation
    test_df_sim = df_clean.loc[test_mask].copy()
    test_df_sim['PredProb'] = probs
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        count = preds.sum()
        
        # ROI Sim: Back blindly if Pred=1
        bets = test_df_sim[test_df_sim['PredProb'] >= t].copy()
        
        roi = 0.0
        win_rate = 0.0
        avg_price = 0.0
        
        if len(bets) > 0:
            # Win Logic
            bets['Win'] = (pd.to_numeric(bets['Position'], errors='coerce') == 1).astype(int)
            win_rate = bets['Win'].mean() * 100
            avg_price = bets['Price5Min'].mean()
            
            pnl = np.where(bets['Win']==1, (bets['Price5Min']-1)*0.95, -1.0).sum()
            roi = pnl / len(bets) * 100
            
        print(f"{t:<6.1f} | {count:<6} | {prec:>9.1%} | {win_rate:>9.1f}% | {avg_price:>9.2f}  | {roi:>9.2f}%")
        
        if roi > best_roi and count > 50:
            best_roi = roi

    print("="*80)
    
    # Save
    joblib.dump(model, 'models/xgb_v44_steamer.pkl')
    print("\nModel saved to models/xgb_v44_steamer.pkl")

if __name__ == "__main__":
    train_v44_classifier()
