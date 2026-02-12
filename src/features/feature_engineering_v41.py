"""Feature engineering V41 - Used by V44/V45 models.

⚠️ This version (V41) is actively used by predict_v41_tips.py which generates
features for the V44 Back Steamer and V45 Lay Drifter production models.

DO NOT REMOVE - Required for production predictions.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineerV41:
    """
    V41 SUPER MODEL: Merging V36 Context + V40 Killer Speed/Interference.
    """
    
    def calculate_features(self, df):
        print("Engineering V41 Features (Context + Speed + Interference)...")
        
        # 1. PREP
        # -----------------------------------------------------
        numeric_cols = ['Position', 'FinishTime', 'Split', 'BSP', 'Margin', 'Weight', 'Distance', 'Box']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Place' not in df.columns and 'Position' in df.columns:
            df['Place'] = df['Position']
        df['win'] = (df['Place'] == 1).astype(int)
        
        # Sort for rolling
        df = df.sort_values(['GreyhoundID', 'MeetingDate'])
        grouped = df.groupby('GreyhoundID')

        # 2. V40 KILLER FEATURES (Speed/Interference)
        # -----------------------------------------------------
        
        # Dog Age
        if 'DateWhelped' in df.columns and 'MeetingDate' in df.columns:
            df['MeetingDate'] = pd.to_datetime(df['MeetingDate']).dt.tz_localize(None)
            df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce').dt.tz_localize(None)
            df['DogAgeDays'] = (df['MeetingDate'] - df['DateWhelped']).dt.days
        else:
            df['DogAgeDays'] = 0

        # Vacant Box (Vectorized)
        race_boxes = df.groupby('RaceID')['Box'].apply(set).to_dict()
        def check_vacant(row):
            left = row['Box'] - 1
            right = row['Box'] + 1
            occupied = race_boxes.get(row['RaceID'], set())
            has_vacant = 0
            if left >= 1 and left not in occupied: has_vacant = 1
            if right <= 8 and right not in occupied: has_vacant = 1
            return has_vacant
        
        if 'Box' in df.columns:
             df['HasVacantBox'] = df.apply(check_vacant, axis=1)
        else:
             df['HasVacantBox'] = 0

        # RunTimeNorm (Lagged)
        winners = df[df['win'] == 1].copy()
        if not winners.empty:
            daily_stats = winners.groupby(['TrackName', 'Distance', 'MeetingDate'])['FinishTime'].median().reset_index()
            daily_stats.rename(columns={'FinishTime': 'DailyWinTime'}, inplace=True)
            daily_stats = daily_stats.sort_values('MeetingDate')
            
            # Compute rolling median per Track/Distance using transform (avoids apply/reset_index quirks)
            daily_stats = daily_stats.sort_values(['TrackName', 'Distance', 'MeetingDate'])
            
            # Shift and rolling median (look-back only to prevent leakage)
            daily_stats['TrackDistMedian'] = daily_stats.groupby(['TrackName', 'Distance'])['DailyWinTime'].transform(
                lambda x: x.shift(1).rolling(365, min_periods=1).median()
            )
            
            # Keep only the columns we need for merging
            benchmarks = daily_stats[['TrackName', 'Distance', 'MeetingDate', 'TrackDistMedian']].copy()
            
            # Ensure MeetingDate types match
            benchmarks['MeetingDate'] = pd.to_datetime(benchmarks['MeetingDate']).dt.tz_localize(None)
            if 'MeetingDate' in df.columns:
                df['MeetingDate'] = pd.to_datetime(df['MeetingDate']).dt.tz_localize(None)

            df = df.merge(benchmarks, on=['TrackName', 'Distance', 'MeetingDate'], how='left')
            df = df.sort_values('MeetingDate')
            df['TrackDistMedian'] = df.groupby(['TrackName', 'Distance'])['TrackDistMedian'].ffill()
            df['RunTimeNorm'] = df['TrackDistMedian'] / df['FinishTime']
            
            # LAG IT (Strict Leakage Prevention)
            df = df.sort_values(['GreyhoundID', 'MeetingDate'])
            grouped = df.groupby('GreyhoundID')
            df['RunTimeNorm_Lag1'] = grouped['RunTimeNorm'].shift(1)
            df['RunTimeNorm_Lag3'] = grouped['RunTimeNorm'].shift(1).rolling(3, min_periods=1).mean()
        else:
            df['RunTimeNorm_Lag1'] = 0
            df['RunTimeNorm_Lag3'] = 0

        # 3. V36 CONTEXTUAL FEATURES (Trainer/Box/Track)
        # -----------------------------------------------------
        # Need re-sort for contextual aggregation
        df = df.sort_values('MeetingDate')
        
        # Trainer @ Track (Rolling)
        tt_group = df.groupby(['TrainerID', 'TrackName', 'MeetingDate'])
        tt_daily = tt_group['win'].agg(['sum', 'count']).reset_index().sort_values(['TrainerID', 'TrackName', 'MeetingDate'])
        
        g = tt_daily.groupby(['TrainerID', 'TrackName'])
        tt_daily['Trainer_Track_Wins'] = g['sum'].cumsum().shift(1).fillna(0)
        tt_daily['Trainer_Track_Runs'] = g['count'].cumsum().shift(1).fillna(0)
        
        df = df.merge(tt_daily[['TrainerID', 'TrackName', 'MeetingDate', 'Trainer_Track_Wins', 'Trainer_Track_Runs']], 
                     on=['TrainerID', 'TrackName', 'MeetingDate'], how='left')
        
        df['Trainer_Track_Rate'] = (df['Trainer_Track_Wins'] / df['Trainer_Track_Runs']).fillna(0)

        # Box @ Track (Bias)
        bt_group = df.groupby(['TrackName', 'Box', 'MeetingDate'])
        bt_daily = bt_group['win'].agg(['sum', 'count']).reset_index().sort_values(['TrackName', 'Box', 'MeetingDate'])
        
        gb = bt_daily.groupby(['TrackName', 'Box'])
        bt_daily['Box_Track_Wins'] = gb['sum'].cumsum().shift(1).fillna(0)
        bt_daily['Box_Track_Runs'] = gb['count'].cumsum().shift(1).fillna(0)
        
        df = df.merge(bt_daily[['TrackName', 'Box', 'MeetingDate', 'Box_Track_Wins', 'Box_Track_Runs']], 
                     on=['TrackName', 'Box', 'MeetingDate'], how='left')
                     
        df['Box_Track_Rate'] = (df['Box_Track_Wins'] / df['Box_Track_Runs']).fillna(0.125)

        # 4. BASIC LAGS (Form)
        # -----------------------------------------------------
        df = df.sort_values(['GreyhoundID', 'MeetingDate'])
        grouped = df.groupby('GreyhoundID')
        
        # Expanding Win Rate (Dog Ability)
        df['Dog_Win_Rate'] = grouped['win'].transform(lambda x: x.shift().expanding().mean()).fillna(0)
        
        # Recent Form
        df['Place_Lag1'] = grouped['Place'].shift(1)
        df['Split_Lag1'] = grouped['Split'].shift(1)

        # Cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        return df

    def get_feature_list(self):
        return [
            # Physics / Topaz
            'RunTimeNorm_Lag3', 'HasVacantBox', 'DogAgeDays',
            # Context / V36
            'Trainer_Track_Rate', 'Trainer_Track_Runs',
            'Box_Track_Rate',
            # Base Ability
            'Dog_Win_Rate', 'Place_Lag1', 'Split_Lag1',
            # Meta
            'Box', 'Distance', 'Weight'
        ]
