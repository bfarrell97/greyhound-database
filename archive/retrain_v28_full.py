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

def train_model():
    print("="*70)
    print("V28 - FULL RETRAINING (2020-Present) - PRODUCTION")
    print("="*70)
    
    # ===== 1. LOAD DATA =====
    print("\n[1/6] Loading data from database...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID as FastTrack_RaceId, ge.GreyhoundID as FastTrack_DogId,
        ge.Box, ge.Position as Place, ge.FinishTime as RunTime, ge.Split as SplitMargin,
        ge.BSP as StartPrice, ge.PrizeMoney as Prizemoney,
        r.Distance, t.TrackName as Track, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2019-01-01'
    AND ge.FinishTime > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # ===== 2. FEATURE ENGINEERING (TUTORIAL MATCH) =====
    print("\n[2/6] Feature Engineering...")
    
    # 2.0 CLEAN PLACE (CRITICAL FIX)
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    df = df.dropna(subset=['Place'])
    df = df[df['Place'] > 0]
    
    # 2.0.1 SORT BY DATE (CRITICAL for Expanding Mean)
    df = df.sort_values('date_dt')
    
    # 2.0.2 DATA CLEANING
    df['StartPrice'] = df['StartPrice'].replace([np.inf, -np.inf], np.nan)
    
    # 2.1 Calculate Median Winner Time per Track/Distance
    win_results = df[df['Place'] == 1]
    median_win_time = win_results.groupby(['Track', 'Distance'])['RunTime'].median().reset_index().rename(columns={'RunTime': 'RunTime_median'})
    median_win_split = win_results.groupby(['Track', 'Distance'])['SplitMargin'].median().reset_index().rename(columns={'SplitMargin': 'SplitMargin_median'})
    
    df = df.merge(median_win_time, on=['Track', 'Distance'], how='left')
    df = df.merge(median_win_split, on=['Track', 'Distance'], how='left')
    
    # 2.2 Calculate Track Speed Index (Tutorial)
    # MedianWinTime / Distance -> Scaled
    median_win_time['speed_index'] = median_win_time['RunTime_median'] / median_win_time['Distance']
    median_win_time['speed_index'] = MinMaxScaler().fit_transform(median_win_time[['speed_index']])
    
    # Merge speed_index back to dogs (Tutorial does this via merge)
    df = df.merge(median_win_time[['Track', 'Distance', 'speed_index']], on=['Track', 'Distance'], how='left')
    
    # 2.3 Normalise Time Comparison (Tutorial)
    # (Median / RunTime) -> Higher is better
    df['RunTime_norm'] = (df['RunTime_median'] / df['RunTime']).clip(0.9, 1.1)
    df['RunTime_norm'] = MinMaxScaler().fit_transform(df[['RunTime_norm']])
    
    df['SplitMargin_norm'] = (df['SplitMargin_median'] / df['SplitMargin']).clip(0.9, 1.1)
    # Handle NaN split
    df['SplitMargin_norm'] = df['SplitMargin_norm'].fillna(df['SplitMargin_norm'].mean())
    df['SplitMargin_norm'] = MinMaxScaler().fit_transform(df[['SplitMargin_norm']])
    
    # 2.4 Box Winning Percentage (Safe Version)
    # Tutorial does global mean. We use expanding mean shifted (No Leakage)
    df['win'] = np.where(df['Place'] == 1, 1, 0)
    df['box_win_percent'] = df.groupby(['Track', 'Distance', 'Box'])['win'].transform(lambda x: x.shift().expanding().mean())
    df['box_win_percent'] = df['box_win_percent'].fillna(0.125)
    
    # 2.5 Normalise others
    df['Prizemoney_norm'] = np.log1p(df['Prizemoney']) / 12 # Tutorial divides by 12
    df['Place_inv'] = (1 / df['Place']).fillna(0)
    df['Place_log'] = np.log1p(df['Place']).fillna(0)
    df['RunSpeed'] = (df['Distance'] / df['RunTime']).fillna(0)
    
    # ===== 3. ROLLING WINDOWS =====
    print("\n[3/6] Calculating Rolling Windows...")
    df = df.sort_values(['FastTrack_DogId', 'date_dt'])
    
    features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'RunSpeed']
    windows = ['28D', '91D', '365D']
    aggregates = ['mean', 'min', 'max', 'std', 'median']
    
    # Base features from Tutorial
    feature_cols = ['speed_index', 'box_win_percent']
    
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
    print(f"Date Range: {train_data['date_dt'].min()} to {train_data['date_dt'].max()}")
    
    save_path = 'models/autogluon_v28_full'
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
        
    # ===== 5. TRAIN AUTOGLUON =====
    # ===== 5. TRAIN AUTOGLUON =====
    print(f"\n[5/6] Training AutoGluon (High Quality - {TIME_LIMIT}s)...")
    # MEMORY OPTIMIZATION:
    predictor = TabularPredictor(label='win', path=save_path, eval_metric='log_loss').fit(
        train_data[feature_cols + ['win']],
        presets='best_quality',
        time_limit=TIME_LIMIT,
        num_bag_folds=5,   # Reduced from 8
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
    train_model()
