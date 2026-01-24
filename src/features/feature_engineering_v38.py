import pandas as pd
import numpy as np
import re

class FeatureEngineerV38:
    def __init__(self):
        self.mappings = {}

    def parse_in_run(self, val):
        """
        Parses 'InRun' string (e.g. '121', '888', 'M23w').
        Returns Early Speed (1st pos) and Late Speed (diff).
        """
        if not isinstance(val, str):
            return np.nan
            
        # Extract digits
        digits = [int(d) for d in val if d.isdigit()]
        if not digits:
            return np.nan
            
        # Early Speed = First recorded position
        early = digits[0]
        
        return early

    def engineer_features(self, df):
        """
        V38 Feature Engineering
        Includes V37 features + Beyer + InRun features.
        """
        print("Sorting data for time-series integrity...")
        # Sort key
        sort_cols = ['date_dt', 'RaceID', 'RawBox']
        if 'RaceTime' in df.columns:
            sort_cols.insert(1, 'RaceTime')
            
        df = df.sort_values(sort_cols)

        # Pre-calc Win
        df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
        df['win'] = (df['Place'] == 1).astype(int)
        
        # Beyer handling
        # Ensure numeric
        df['BeyerSpeedFigure'] = pd.to_numeric(df['BeyerSpeedFigure'], errors='coerce')
        # Fill missing Beyer with... 0? Or Avg? 
        # For Lags, 0 is fine (indicates no data). 
        # Ideally we want the model to know "Unknown Speed" vs "Slow Speed".
        # 0 is strictly speaking "Very Slow". 
        # But XGBoost handles missing values if we leave them as NaN?
        # Let's leave as NaN for now for calculation, fill later.
        
        # 1. LAG FEATURES (Dog Specific)
        # -----------------------------
        print("Calculating Dog Lags (Split, Place, Beyer)...")
        # Reset index to ensure 0..N clean index
        df = df.sort_values(['GreyhoundID', 'date_dt']).reset_index(drop=True)
        
        g_dog = df.groupby('GreyhoundID')
        
        print("  - Split Lags")
        df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
        df['Split_Lag1'] = g_dog['Split'].shift(1)
        
        print("  - Beyer Lags")
        df['Beyer_Lag1'] = g_dog['BeyerSpeedFigure'].shift(1)
        df['Beyer_Lag2'] = g_dog['BeyerSpeedFigure'].shift(2)
        df['Beyer_Lag3'] = g_dog['BeyerSpeedFigure'].shift(3)
        
        print("  - Beyer Avg")
        # Rolling returns MultiIndex (Group, Index). Must drop Group level.
        # Shift must happen AFTER rolling mean (to lag the average).
        # But wait, shifting a MultiIndex might shift across groups if not careful?
        # Safe way: groupby -> shift -> rolling? No, rolling needs history.
        # rolling -> mean -> shift? 
        # Easier: Compute rolling mean, THEN GroupBy again to Shift? Or just shift the result?
        # If we shift the result, we might shift Group A's last value into Group B's first value if we ignore groups.
        # But the result has MultiIndex.
        
        # Correct pattern:
        # Calculate rolling, reset level 0 to match df index.
        # Ensure 'df' index alignment.
        # Then we still need to SHIFT validly. 
        # Shifting the whole column by 1 is wrong (leaks across dogs).
        # We need `g_dog['Beyer_Avg'].shift(1)`.
        
        # Two step:
        # 1. Calculate Rolling (Current)
        rolling_mean = g_dog['BeyerSpeedFigure'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        # Assign temporarily
        df['Beyer_Avg_3_Curr'] = rolling_mean
        # 2. Shift via GroupBy
        # Need to re-group because 'roll_mean' is just a column now.
        # Re-using g_dog is tricky if df changed? No, df just got a new column.
        # Group again safely.
        df['Beyer_Avg_3'] = df.groupby('GreyhoundID')['Beyer_Avg_3_Curr'].shift(1)
        df.drop(columns=['Beyer_Avg_3_Curr'], inplace=True)
        
        # 2. IN_RUN PARSING (Recovery/EarlySpeed)
        # ---------------------------------------
        print("Parsing InRun data...")
        df['InRun'] = df['InRun'].astype(str)
        
        print("  - Extracting digits")
        extracted = df['InRun'].str.extract(r'(\d)')
        df['EarlyPos'] = extracted[0].astype(float)
        
        print("  - Calculating Strength")
        df['LateStrength'] = df['Place'] - df['EarlyPos']
        
        g_dog = df.groupby('GreyhoundID')
        
        print("  - Lagging Early Speed")
        df['EarlyPos_Lag1'] = g_dog['EarlyPos'].shift(1)
        df['LateStrength_Lag1'] = g_dog['LateStrength'].shift(1)
        
        # Rolling Early Speed
        rolling_speed = g_dog['EarlyPos'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        df['Avg_ES_Curr'] = rolling_speed
        df['Avg_Early_Speed'] = df.groupby('GreyhoundID')['Avg_ES_Curr'].shift(1)
        df.drop(columns=['Avg_ES_Curr'], inplace=True)
        
        # 3. TRAINER FORM
        # ----------------
        print("Calculating Trainer Form...")
        df = df.sort_values(['date_dt', 'RaceID'])
        
        trainer_daily = df.groupby(['TrainerID', 'date_dt']).agg(
            daily_wins=('win', 'sum'),
            daily_runs=('win', 'count')
        ).reset_index().sort_values(['TrainerID', 'date_dt'])
        
        trainer_daily['cum_wins'] = trainer_daily.groupby('TrainerID')['daily_wins'].cumsum().shift(1).fillna(0)
        trainer_daily['cum_runs'] = trainer_daily.groupby('TrainerID')['daily_runs'].cumsum().shift(1).fillna(0)
        
        df = df.merge(trainer_daily[['TrainerID', 'date_dt', 'cum_wins', 'cum_runs']], on=['TrainerID', 'date_dt'], how='left')
        df['Trainer_Win_Rate'] = (df['cum_wins'] / df['cum_runs']).fillna(0)
        
        # 4. TRACK SPEED PROFILE (Beyer @ Track)
        # --------------------------------------
        print("Calculating Track Proficiency...")
        # Avg Beyer for this Dog at this Track (Prior to today)
        
        # Sort and clean index
        df = df.sort_values(['GreyhoundID', 'RawTrack', 'date_dt']).reset_index(drop=True)
        g_dog_track = df.groupby(['GreyhoundID', 'RawTrack'])
        
        # Expanding Mean
        # Result has MultiIndex (Dog, Track, Index). Reset to match df index.
        expanding_mean = g_dog_track['BeyerSpeedFigure'].expanding().mean().reset_index(level=[0, 1], drop=True)
        
        df['Beyer_Track_Avg_Curr'] = expanding_mean
        
        # Shift to avoid leakage
        # Re-grouping needed? g_dog_track is valid on new df.
        # But to be safe, group again on the modified df (it has the new column).
        df['Beyer_Track_Avg'] = df.groupby(['GreyhoundID', 'RawTrack'])['Beyer_Track_Avg_Curr'].shift(1)
        df.drop(columns=['Beyer_Track_Avg_Curr'], inplace=True)
        
        df['Dog_Track_Runs'] = df.groupby(['GreyhoundID', 'RawTrack']).cumcount()
        
        # Fill NaNs for features
        features = [
            'Trainer_Win_Rate',
            'Split_Lag1',
            'Beyer_Lag1', 'Beyer_Lag2', 'Beyer_Avg_3',
            'Beyer_Track_Avg',
            'EarlyPos_Lag1', 'LateStrength_Lag1', 'Avg_Early_Speed',
            'Dog_Track_Runs'
        ]
        
        df[features] = df[features].fillna(-1) 
        
        return df, features
