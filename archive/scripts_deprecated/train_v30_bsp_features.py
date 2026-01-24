"""
V30 - BSP Market Features Model
================================
Builds on V28 rolling window features, adding historical BSP (log-transformed)
as an additional rolling window feature.

Key Features:
- V28 rolling windows (RunTime_norm, SplitMargin_norm, Place_inv, Place_log, Prizemoney_norm)
- NEW: BSP_log rolling windows (mean, std over 28D/91D/365D)
- Box win percentages, Speed index
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import itertools
import warnings
warnings.filterwarnings('ignore')

def train_v30():
    print("="*70)
    print("V30 - BSP MARKET FEATURES MODEL")
    print("="*70)
    
    # ===== 1. LOAD DATA FROM DATABASE =====
    print("\n[1/7] Loading data from database...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    query = """
    SELECT 
        ge.EntryID,
        ge.RaceID as FastTrack_RaceId,
        ge.GreyhoundID as FastTrack_DogId,
        ge.Box,
        ge.Weight,
        ge.Position as Place,
        ge.Margin as Margin1,
        ge.FinishTime as RunTime,
        ge.Split as SplitMargin,
        ge.BSP as StartPrice,
        ge.PrizeMoney as Prizemoney,
        r.Distance,
        t.TrackName as Track,
        rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2019-01-01'
      AND ge.Position IS NOT NULL 
      AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    dog_results = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(dog_results):,} entries")
    
    # ===== 2. CLEANSE AND NORMALISE =====
    print("\n[2/7] Cleansing and normalising data...")
    
    # Convert types
    dog_results['date_dt'] = pd.to_datetime(dog_results['date_dt'])
    dog_results['Place'] = pd.to_numeric(dog_results['Place'], errors='coerce')
    dog_results['RunTime'] = pd.to_numeric(dog_results['RunTime'], errors='coerce')
    dog_results['SplitMargin'] = pd.to_numeric(dog_results['SplitMargin'], errors='coerce')
    dog_results['StartPrice'] = pd.to_numeric(dog_results['StartPrice'], errors='coerce')
    dog_results['Prizemoney'] = pd.to_numeric(dog_results['Prizemoney'], errors='coerce').fillna(0)
    dog_results['Distance'] = pd.to_numeric(dog_results['Distance'], errors='coerce')
    dog_results['Box'] = pd.to_numeric(dog_results['Box'], errors='coerce').fillna(0).astype(int)
    dog_results['Weight'] = pd.to_numeric(dog_results['Weight'], errors='coerce')
    
    # Discard invalid entries
    dog_results = dog_results[dog_results['Place'].notna()]
    dog_results = dog_results[dog_results['Box'] > 0]
    
    # Win flag
    dog_results['win'] = (dog_results['Place'] == 1).astype(int)
    
    # StartPrice probability (normalized per race)
    dog_results['StartPrice_probability'] = (1 / dog_results['StartPrice']).fillna(0)
    dog_results['StartPrice_probability'] = dog_results.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x / x.sum())
    
    # Normalise values (as per tutorial)
    dog_results['Prizemoney_norm'] = np.log10(dog_results['Prizemoney'] + 1) / 12
    dog_results['Place_inv'] = (1 / dog_results['Place']).fillna(0)
    dog_results['Place_log'] = np.log10(dog_results['Place'] + 1).fillna(0)
    dog_results['RunSpeed'] = (dog_results['RunTime'] / dog_results['Distance']).fillna(0)
    
    # NEW: Log-transform BSP for rolling window features
    dog_results['BSP_log'] = np.log(dog_results['StartPrice'].clip(lower=1.01)).fillna(0)
    
    # ===== 3. CALCULATE TRACK REFERENCE TIMES =====
    print("\n[3/7] Calculating Track/Distance reference times...")
    
    win_results = dog_results[dog_results['win'] == 1]
    
    # Median winner time per track/distance
    median_win_time = win_results[win_results['RunTime'] > 0].groupby(['Track', 'Distance'])['RunTime'].median().reset_index()
    median_win_time.columns = ['Track', 'Distance', 'RunTime_median']
    
    # Median winner split per track/distance
    median_win_split = win_results[win_results['SplitMargin'] > 0].groupby(['Track', 'Distance'])['SplitMargin'].median().reset_index()
    median_win_split.columns = ['Track', 'Distance', 'SplitMargin_median']
    
    # Speed index
    median_win_time['speed_index'] = median_win_time['RunTime_median'] / median_win_time['Distance']
    median_win_time['speed_index'] = MinMaxScaler().fit_transform(median_win_time[['speed_index']])
    
    # Merge reference times
    dog_results = dog_results.merge(median_win_time[['Track', 'Distance', 'RunTime_median', 'speed_index']], 
                                     on=['Track', 'Distance'], how='left')
    dog_results = dog_results.merge(median_win_split, on=['Track', 'Distance'], how='left')
    
    # Normalise time comparison
    dog_results['RunTime_norm'] = (dog_results['RunTime_median'] / dog_results['RunTime']).clip(0.9, 1.1)
    dog_results['RunTime_norm'] = MinMaxScaler().fit_transform(dog_results[['RunTime_norm']])
    
    dog_results['SplitMargin_norm'] = (dog_results['SplitMargin_median'] / dog_results['SplitMargin']).clip(0.9, 1.1)
    dog_results['SplitMargin_norm'] = MinMaxScaler().fit_transform(dog_results[['SplitMargin_norm']])
    
    # ===== 4. BOX WIN PERCENTAGES =====
    print("\n[4/7] Calculating Box win percentages...")
    box_win_percent = dog_results.groupby(['Track', 'Distance', 'Box'])['win'].mean().reset_index()
    box_win_percent.columns = ['Track', 'Distance', 'Box', 'box_win_percent']
    dog_results = dog_results.merge(box_win_percent, on=['Track', 'Distance', 'Box'], how='left')
    
    # ===== 5. ROLLING WINDOW FEATURES =====
    print("\n[5/7] Generating rolling window features (this may take a while)...")
    
    dataset = dog_results.copy()
    dataset = dataset.set_index(['FastTrack_DogId', 'date_dt']).sort_index()
    
    rolling_windows = ['28D', '91D', '365D']
    features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'BSP_log']
    aggregates = ['min', 'max', 'mean', 'median', 'std']
    feature_cols = ['speed_index', 'box_win_percent']
    
    for rolling_window in rolling_windows:
        print(f'  Processing rolling window {rolling_window}...')
        
        rolling_result = (
            dataset
            .reset_index(level=0)
            .groupby('FastTrack_DogId')[features]
            .rolling(rolling_window)
            .agg(aggregates)
            .groupby(level=0)
            .shift(1)
        )
        
        agg_features_cols = [f'{f}_{a}_{rolling_window}' for f, a in itertools.product(features, aggregates)]
        dataset[agg_features_cols] = rolling_result
        feature_cols.extend(agg_features_cols)
    
    dataset.fillna(0, inplace=True)
    
    # ===== 6. PREPARE MODEL DATA =====
    print("\n[6/7] Preparing train/test split...")
    
    model_df = dataset.reset_index()
    feature_cols = list(set(feature_cols))
    
    # Only keep data after first year (need history)
    model_df = model_df[model_df['date_dt'] >= '2020-01-01']
    model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'StartPrice_probability'] + feature_cols]
    
    # Drop races with missing features
    races_exclude = model_df[model_df.isnull().any(axis=1)]['FastTrack_RaceId'].drop_duplicates()
    model_df = model_df[~model_df['FastTrack_RaceId'].isin(races_exclude)]
    
    print(f"Dataset size: {len(model_df):,} rows, {len(feature_cols)} features")
    
    # Split - use 80/20 split based on date
    split_date = model_df['date_dt'].quantile(0.8)
    train_data = model_df[model_df['date_dt'] < split_date].reset_index(drop=True)
    test_data = model_df[model_df['date_dt'] >= split_date].reset_index(drop=True)
    
    print(f"Train: {len(train_data):,}, Test: {len(test_data):,}")
    
    # ===== 7. TRAIN AUTOGLUON =====
    print("\n[7/7] Training AutoGluon...")
    from autogluon.tabular import TabularPredictor
    
    save_path = 'models/autogluon_v30_bsp'
    
    predictor = TabularPredictor(
        label='win',
        path=save_path,
        problem_type='binary',
        eval_metric='log_loss'
    ).fit(
        train_data[feature_cols + ['win']],
        presets='high_quality',
        ag_args_fit={'save_bag_folds': True, 'ag.max_memory_usage_ratio': 1.2}
    )
    
    # ===== EVALUATION =====
    print("\n" + "="*70)
    print("V30 EVALUATION REPORT (with BSP features)")
    print("="*70)
    
    # Get probabilities
    probs = predictor.predict_proba(test_data[feature_cols])
    test_data = test_data.copy()
    test_data['prob_model'] = probs[1]
    
    # Normalise probabilities per race
    test_data['prob_model'] = test_data.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())
    
    # 1. Strike Rate
    test_dataset_size = test_data['FastTrack_RaceId'].nunique()
    predicted_winners = test_data.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x == x.max())
    model_strike_rate = len(test_data[(predicted_winners == True) & (test_data['win'] == 1)]) / test_dataset_size
    
    sp_predicted_winners = test_data.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x == x.max())
    sp_strike_rate = len(test_data[(sp_predicted_winners == True) & (test_data['win'] == 1)]) / test_dataset_size
    
    print(f"Starting Price Strike Rate: {sp_strike_rate:.2%}")
    print(f"Model Strike Rate:          {model_strike_rate:.2%}")
    
    # 2. Brier Score
    from sklearn.metrics import brier_score_loss
    brier_sp = brier_score_loss(test_data['win'], test_data['StartPrice_probability'])
    brier_model = brier_score_loss(test_data['win'], test_data['prob_model'])
    
    print(f"\nStarting Price Brier Score: {brier_sp:.6f}")
    print(f"Model Brier Score:          {brier_model:.6f}")
    
    # 3. Price Accuracy (Rated Price vs BSP)
    test_data['RatedPrice'] = 1 / test_data['prob_model']
    valid = test_data[(test_data['StartPrice'] > 1) & (test_data['StartPrice'] < 50)].copy()
    valid['PctError'] = np.abs(valid['RatedPrice'] - valid['StartPrice']) / valid['StartPrice']
    mape = valid['PctError'].mean() * 100
    corr = valid[['RatedPrice', 'StartPrice']].corr().iloc[0,1]
    
    print(f"\nPrice Accuracy (vs BSP):")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation: {corr:.4f}")
    
    # 4. Betting Simulation (with 10% commission)
    COMMISSION = 0.10
    print(f"\nBetting Validation (Value > 20%, {COMMISSION*100:.0f}% commission):")
    bets = valid[valid['StartPrice'] > valid['RatedPrice'] * 1.2].copy()
    # Commission applies to profit on winning bets only
    bets['Profit'] = np.where(bets['win']==1, (bets['StartPrice'] - 1) * (1 - COMMISSION), -1)
    roi = bets['Profit'].sum() / len(bets) * 100 if len(bets) > 0 else 0
    wins = bets['win'].sum()
    strike_rate = wins / len(bets) * 100 if len(bets) > 0 else 0
    print(f"  Bets: {len(bets):,}")
    print(f"  Wins: {wins:,} ({strike_rate:.1f}%)")
    print(f"  ROI (after commission): {roi:.2f}%")

if __name__ == "__main__":
    train_v30()
