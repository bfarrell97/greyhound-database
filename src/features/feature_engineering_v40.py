"""Legacy feature engineering (V40) - DEPRECATED.

⚠️ Older version kept for reference. Use feature_engineering.py instead.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineerV40:
    """
    V40 Hybrid: Mering V33 Base Ability with V39 Topaz Speed/Interference.
    """
    
    def calculate_features(self, df):
        print("Engineering V40 Features (Hybrid)...")
        # Ensure Types
        numeric_cols = ['Position', 'FinishTime', 'Split', 'BSP', 'Margin', 'Weight', 'Distance', 'Box']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Handle 'Place' alias
        if 'Place' not in df.columns and 'Position' in df.columns:
            df['Place'] = df['Position']

        df['win'] = (df['Place'] == 1).astype(int)
        
        # Sort for rolling
        df = df.sort_values(['GreyhoundID', 'MeetingDate'])
        grouped = df.groupby('GreyhoundID')
        
        # -----------------------------------------------------------
        # 1. V33 BASE FEATURES (Class, Form, Base Speed)
        # -----------------------------------------------------------
        # Strike Rate (Win %) - Expanding
        df['SR_avg'] = grouped['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
        
        # Speed (Raw)
        df['RunSpeed'] = df['Distance'] / df['FinishTime']
        df['RunSpeed_avg'] = grouped['RunSpeed'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
        
        # Lags (Place, Time, Split, BSP)
        # We limit to 3 for efficiency in V40 (V33 used 10 for place, but 3 is cleaner)
        for i in range(1, 4):
            df[f'Place_Lag{i}'] = grouped['Place'].shift(i)
            df[f'RunTime_Lag{i}'] = grouped['FinishTime'].shift(i)
            df[f'Split_Lag{i}'] = grouped['Split'].shift(i)
            df[f'BSP_Lag{i}'] = grouped['BSP'].shift(i)

        # -----------------------------------------------------------
        # 2. V39 TOPAZ FEATURES (Interference, Norm Speed, Age)
        # -----------------------------------------------------------
        
        # Dog Age (Experience)
        if 'DateWhelped' in df.columns and 'MeetingDate' in df.columns:
            df['MeetingDate'] = pd.to_datetime(df['MeetingDate']).dt.tz_localize(None)
            df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce').dt.tz_localize(None)
            df['DogAgeDays'] = (df['MeetingDate'] - df['DateWhelped']).dt.days
        else:
            df['DogAgeDays'] = 0
            
        # Vacant Box (Interference)
        # Needs Race Geometry. We assume df has 'Box' and 'RaceID'.
        # This is expensive to calc per race if not pre-calc.
        # fast approximation: Group by RaceID, get set of Boxes.
        # (Implementing optimized vectorized approach)
        
        # Map RaceID -> Set of Boxes
        race_boxes = df.groupby('RaceID')['Box'].apply(set).to_dict()
        
        def check_vacant(row):
            left = row['Box'] - 1
            right = row['Box'] + 1
            occupied = race_boxes.get(row['RaceID'], set())
            
            # Box 1 has no left, Box 8 has no right
            # If left is valid (>=1) and NOT in occupied -> Vacant
            # If right is valid (<=8) and NOT in occupied -> Vacant
            # Logic: Has ANY vacant neighbor?
            
            has_vacant = 0
            if left >= 1 and left not in occupied: has_vacant = 1
            if right <= 8 and right not in occupied: has_vacant = 1
            return has_vacant

        # Only apply if we have Box data
        if 'Box' in df.columns:
             df['HasVacantBox'] = df.apply(check_vacant, axis=1) # Slow but exact
        else:
             df['HasVacantBox'] = 0
             
        # RunTimeNorm (Normalized Speed) - LAGGED ONLY
        # 1. Calc Benchmark (Daily shifted rolling median)
        winners = df[df['win'] == 1].copy()
        if not winners.empty:
            daily_stats = winners.groupby(['TrackName', 'Distance', 'MeetingDate'])['FinishTime'].median().reset_index()
            daily_stats.rename(columns={'FinishTime': 'DailyWinTime'}, inplace=True)
            daily_stats = daily_stats.sort_values('MeetingDate')
            
            def calc_bench(g):
                g = g.set_index('MeetingDate')
                return g['DailyWinTime'].shift(1).rolling('365D', min_periods=1).median()
                
            benchmarks = daily_stats.groupby(['TrackName', 'Distance']).apply(calc_bench).reset_index()
            benchmarks.rename(columns={'DailyWinTime': 'TrackDistMedian'}, inplace=True)
            
            # Merge
            df = df.merge(benchmarks, on=['TrackName', 'Distance', 'MeetingDate'], how='left')
            # FFill for inference
            df = df.sort_values('MeetingDate')
            df['TrackDistMedian'] = df.groupby(['TrackName', 'Distance'])['TrackDistMedian'].ffill()
            
            # Calc Raw Norm (For history)
            df['RunTimeNorm'] = df['TrackDistMedian'] / df['FinishTime']
            
            # Lag it (The Feature)
            # Re-sort by Dog
            df = df.sort_values(['GreyhoundID', 'MeetingDate'])
            grouped = df.groupby('GreyhoundID')
            
            df['RunTimeNorm_Lag1'] = grouped['RunTimeNorm'].shift(1)
            df['RunTimeNorm_Lag3'] = grouped['RunTimeNorm'].shift(1).rolling(3, min_periods=1).mean()
        else:
            df['RunTimeNorm_Lag1'] = 0
            df['RunTimeNorm_Lag3'] = 0
            
        # Cleanup
        df = df.replace([np.inf, -np.inf], 0)
            
        return df

    def get_feature_list(self):
        return [
            'Box', 'Distance', 'Weight', # Basic
            'SR_avg', 'RunSpeed_avg',    # V33 Base
            'Place_Lag1', 'RunTime_Lag1', 'Split_Lag1', 'BSP_Lag1', # V33 Lags
            'DogAgeDays', 'HasVacantBox', # V39 Static
            'RunTimeNorm_Lag1', 'RunTimeNorm_Lag3' # V39 Killer
        ]
