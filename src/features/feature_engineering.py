"""Feature engineering for greyhound racing predictions.

Calculates predictive features from historical race data including:
- Lag features (dog-specific recent performance)
- Trainer form (win rates, runs)
- Box bias (track/distance/box win probabilities)
- Track experience (dog familiarity with track)

Features are calculated in a time-aware manner to prevent data leakage
(no future information used in training/prediction).

Example:
    >>> from src.features.feature_engineering import FeatureEngineerV37
    >>> engineer = FeatureEngineerV37()
    >>> df_with_features, feature_names = engineer.engineer_features(raw_df)
    >>> print(f"Created {len(feature_names)} features")
    Created 9 features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class FeatureEngineerV37:
    """Feature engineering for greyhound racing ML models.
    
    Generates time-series features from historical race data with strict
    temporal ordering to prevent data leakage. Uses daily aggregations
    and lag shifts to ensure no future information is used.
    
    Features Generated:
        - Split_Lag1, Split_Lag2: Dog's recent split times
        - Place_Lag1: Dog's last finishing position
        - RunSpeed_Lag1: Dog's recent finish time
        - Trainer_Win_Rate: Trainer's cumulative win rate
        - Trainer_Runs_Life: Trainer's total career runs
        - Box_Win_Prob: Win probability for box/track/distance
        - Box_History_Runs: Historical runs from this box
        - Dog_Track_Runs: Number of times dog has run at this track
    
    Attributes:
        mappings (dict): Internal mappings (reserved for future use)
    
    Example:
        >>> engineer = FeatureEngineerV37()
        >>> df, features = engineer.engineer_features(raw_data)
        >>> print(features)
        ['Trainer_Win_Rate', 'Box_Win_Prob', 'Split_Lag1', ...]
    """

    def __init__(self) -> None:
        """Initialize feature engineer."""
        self.mappings: dict = {}

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate features from raw racing data.
        
        Performs time-aware feature engineering with strict temporal ordering.
        Sorts data by date/time, calculates lag features, trainer form, box bias,
        and track experience.
        
        Args:
            df: Raw racing DataFrame with columns:
                - date_dt: Race date (datetime)
                - RaceTime: Race time (HH:MM:SS string)
                - RaceID: Unique race identifier
                - GreyhoundID: Dog identifier
                - TrainerID: Trainer identifier
                - RawTrack: Track name
                - Distance: Race distance (meters)
                - RawBox: Starting box number (1-8)
                - Place: Finishing position
                - Split: Split time (seconds)
                - FinishTime: Final time (seconds)
        
        Returns:
            Tuple of (enriched_dataframe, feature_names_list)
        
        Example:
            >>> raw = load_historical_data()
            >>> df_features, features = engineer.engineer_features(raw)
            >>> X = df_features[features]
            >>> y = df_features['win']
        """
        """
        Main entry point. 
        df: DataFrame containing raw racing data
        """
        print("Sorting data for time-series integrity...")
        # Sort by Date then RaceID (proxy for time)
        # Ensure RaceTime is datetime
        if 'RaceTime' in df.columns:
            # RaceTime is likely string HH:MM:SS
            # We treat it as string sort, or convert to time. 
            # String sort for "HH:MM:SS" works fine for 24h format.
            df = df.sort_values(['date_dt', 'RaceTime', 'RaceID', 'RawBox'])
        else:
             df = df.sort_values(['date_dt', 'RaceID', 'RawBox'])

        # Pre-calc Win
        df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
        df['win'] = (df['Place'] == 1).astype(int)
        
        # 1. LAG FEATURES (Dog Specific)
        # -----------------------------
        print("Calculating Dog Lags...")
        # Group sort
        # Ensure strict sort before shift
        df = df.sort_values(['GreyhoundID', 'date_dt', 'RaceTime', 'RaceID'])
        
        # Split Lags
        df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
        df['Split_Lag1'] = df.groupby('GreyhoundID')['Split'].shift(1)
        df['Split_Lag2'] = df.groupby('GreyhoundID')['Split'].shift(2)
        
        # Position Lags
        df['Place_Lag1'] = df.groupby('GreyhoundID')['Place'].shift(1)
        
        # Speed Lags (Beyer/RunSpeed)
        df['RunSpeed'] = pd.to_numeric(df['FinishTime'], errors='coerce') # Placeholder, real Speed requires Distance
        df['RunSpeed_Lag1'] = df.groupby('GreyhoundID')['RunSpeed'].shift(1)
        
        # 2. TRAINER FORM (EMA)
        # ---------------------
        print("Calculating Trainer Form...")
        # Re-sort for global perspective (Date -> Msg -> Race)
        df = df.sort_values(['date_dt', 'RaceTime', 'RaceID'])
        
        # Rolling stats for Trainer
        # PREVENT INTRA-RACE LEAKAGE:
        # Use Daily Stats (Yesterday's Form) -> Safe.
        
        trainer_daily = df.groupby(['TrainerID', 'date_dt']).agg(
            daily_wins=('win', 'sum'),
            daily_runs=('win', 'count')
        ).reset_index().sort_values(['TrainerID', 'date_dt'])
        
        trainer_daily['cum_wins'] = trainer_daily.groupby('TrainerID')['daily_wins'].cumsum().shift(1).fillna(0)
        trainer_daily['cum_runs'] = trainer_daily.groupby('TrainerID')['daily_runs'].cumsum().shift(1).fillna(0)
        
        df = df.merge(trainer_daily[['TrainerID', 'date_dt', 'cum_wins', 'cum_runs']], on=['TrainerID', 'date_dt'], how='left')
        df['Trainer_Win_Rate'] = (df['cum_wins'] / df['cum_runs']).fillna(0)
        df['Trainer_Runs_Life'] = df['cum_runs']
        
        # 3. BOX BIAS (Track/Dist Specific)
        # ---------------------------------
        print("Calculating Box Bias...")
        # Group by Track, Distance, Box
        
        # Ensure Distance is int or string consistent
        group_cols = ['RawTrack', 'Distance', 'RawBox']
        
        # We need historical stats for this Track/Dist/Box combination.
        # Use Daily aggregation again for safety.
        # Note: 'Distance' in DB might be inconsistent types.
        
        box_daily = df.groupby(['RawTrack', 'Distance', 'RawBox', 'date_dt']).agg(
             daily_wins=('win', 'sum'),
             daily_runs=('win', 'count')
        ).reset_index().sort_values(['date_dt'])
        
        # Group by Track-Dist-Box to cumsum
        g = box_daily.groupby(['RawTrack', 'Distance', 'RawBox'])
        box_daily['Box_History_Wins'] = g['daily_wins'].cumsum().shift(1).fillna(0)
        box_daily['Box_History_Runs'] = g['daily_runs'].cumsum().shift(1).fillna(0)
        
        df = df.merge(box_daily[['RawTrack', 'Distance', 'RawBox', 'date_dt', 'Box_History_Wins', 'Box_History_Runs']], 
                      on=['RawTrack', 'Distance', 'RawBox', 'date_dt'], how='left')
                      
        df['Box_Win_Prob'] = (df['Box_History_Wins'] / df['Box_History_Runs']).fillna(0.125)
        
        # 4. TRACK SPEED PROFILE
        # ----------------------
        print("Calculating Speed Benchmarks...")
        dog_track_group = df.groupby(['GreyhoundID', 'RawTrack', 'Distance'])
        df['Dog_Track_Runs'] = dog_track_group.cumcount()
        # Note: Valid to know "How many times I've run here BEFORE this race".
        # cumcount() starts at 0 for the first run. Correct.
        
        features = [
            'Trainer_Win_Rate', 'Trainer_Runs_Life',
            'Box_Win_Prob', 'Box_History_Runs',
            'Split_Lag1', 'Split_Lag2',
            'Place_Lag1',
            'RunSpeed_Lag1',
            'Dog_Track_Runs'
        ]
        
        # Clean up
        df = df.fillna(0)
        
        return df, features
