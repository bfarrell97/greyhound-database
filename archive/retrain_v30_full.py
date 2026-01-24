"""
V30 - FULL RETRAINING SCHEDULER (2020-Present)
================================================
Prod version of V30 model.
"""
import sqlite3
import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# PARAMETERS
TRAIN_START_DATE = '2020-01-01'
TIME_LIMIT = 10800  # 3 Hours

def train_v30_full():
    print("="*70)
    print("V30 - FULL RETRAINING (2020-Present) - PRODUCTION")
    print("="*70)
    
    # ===== 1. LOAD DATA =====
    print("\n[1/6] Loading data from database...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID as FastTrack_RaceId, ge.GreyhoundID as FastTrack_DogId,
        ge.Box, ge.Weight, ge.TrainerID,
        ge.Position as Place, ge.Margin as Margin1, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.BSP as StartPrice, ge.PrizeMoney as Prizemoney,
        r.Distance, r.Grade, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2019-01-01'
    AND ge.FinishTime > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # ===== 2. FEATURE ENGINEERING =====
    print("[2/6] Feature Engineering...")
    
    # 2.0 CLEAN PLACE
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df = df[df['Place'] > 0]

    # 2.0.1 SORT BY DATE
    df = df.sort_values('date_dt')
    
    # 2.0.2 DATA CLEANING
    df.loc[df['Weight'] < 15, 'Weight'] = np.nan
    df['StartPrice'] = df['StartPrice'].replace([np.inf, -np.inf], np.nan)
    
    # 2.1 Stats
    stats = df[df['Place'] == 1].groupby(['Track', 'Distance'])[['RunTime', 'SplitMargin']].median().reset_index()
    stats.rename(columns={'RunTime': 'MedianWinTime', 'SplitMargin': 'MedianSplit'}, inplace=True)
    df = df.merge(stats, on=['Track', 'Distance'], how='left')
    
    # Speed Index
    df['speed_index'] = df['MedianWinTime'] / df['Distance']
    
    # RunSpeed / Speed
    df['RunSpeed'] = df['Distance'] / df['RunTime']
    df['Speed'] = df['RunSpeed']

    # Normalize RunTime
    df['RunTime_norm'] = df['MedianWinTime'] / df['RunTime']
    
    # Normalize Split
    df['SplitMargin_norm'] = df['MedianSplit'] / df['SplitMargin']
    df['SplitMargin_norm'] = df['SplitMargin_norm'].fillna(1.0) # Default to par

    # 2.3 Targets
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    
    # 2.4 Features
    df['Place_inv'] = 1 / df['Place']
    df['Place_log'] = np.log(df['Place'])
    
    # Box Win Pct (Expanding Mean)
    df['Box_Win_Pct'] = df.groupby(['Track', 'Distance', 'Box'])['win'].transform(lambda x: x.shift().expanding().mean())
    df['Box_Win_Pct'] = df['Box_Win_Pct'].fillna(0.125)

    df['Prizemoney_norm'] = np.log1p(df['Prizemoney'])
    df['BSP_log'] = np.log1p(df['StartPrice'])

    # ===== 3. ROLLING WINDOWS =====
    print("\n[3/6] Calculating Rolling Windows...")
    df = df.sort_values(['FastTrack_DogId', 'date_dt'])
    
    features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'Speed', 'BSP_log']
    windows = ['28D', '91D', '365D']
    aggregates = ['mean', 'min', 'max', 'std', 'median']
    
    feature_cols = ['Box', 'TrainerID', 'Track', 'Distance', 'Grade', 'Weight', 'speed_index', 'Box_Win_Pct']
    
    for w in windows:
        print(f"  Processing {w} window...")
        df_indexed = df.set_index('date_dt')
        for f in features:
            for agg in aggregates:
                col = f'{f}_{agg}_{w}'
                df[col] = df_indexed.groupby('FastTrack_DogId')[f].rolling(w).agg(agg).groupby('FastTrack_DogId').shift(1).reset_index(level=0, drop=True).values
                feature_cols.append(col)

    df.fillna(0, inplace=True)
    
    # ===== 4. TRAIN (2020-Present) =====
    print(f"\n[4/6] Preparing Full Dataset ({TRAIN_START_DATE}-Present)...")
    feature_cols = list(set(feature_cols))
    train_data = df[df['date_dt'] >= TRAIN_START_DATE].copy()
    
    print(f"Full Training Set: {len(train_data):,} rows, {len(feature_cols)} features")
    
    save_path = 'models/autogluon_v30_full'
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
        
    # ===== 5. TRAIN AUTOGLUON =====
    print(f"\n[5/6] Training AutoGluon (High Quality - {TIME_LIMIT}s)...")
    # MEMORY OPTIMIZATION:
    # - Excluded NNs (Too much RAM)
    # - Dynamic Stacking = False
    # - Num Bag Folds = 5 (Standard)
    # - Included: GBM, CAT, XGB, RF, XT
    
    predictor = TabularPredictor(label='win', path=save_path, eval_metric='log_loss').fit(
        train_data[feature_cols + ['win']],
        presets='best_quality',
        time_limit=TIME_LIMIT,
        num_bag_folds=5,   # Reduced from 8 to fit in 32GB RAM
        num_stack_levels=1, 
        dynamic_stacking=False, # Critical for RAM
        excluded_model_types=['NN_TORCH', 'FASTAI', 'KNN'], # Critical for RAM
        ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}
    )
    
    # ===== 6. FEATURE IMPORTANCE =====
    print("\n[6/6] Feature Importance...")
    try:
        fi = predictor.feature_importance(train_data.sample(5000))
        print(fi.head(10))
    except:
        pass
        
    print(f"\nTraining Complete! Model saved to {save_path}")

if __name__ == "__main__":
    train_v30_full()
