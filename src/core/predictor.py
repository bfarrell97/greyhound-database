"""Machine learning prediction wrapper for greyhound race outcomes.

This module provides the RacePredictor class which handles:
- Loading pre-trained XGBoost models (PIR split time, Pace finish time)
- Calculating track/distance benchmarks from historical data
- Feature engineering (lag features, rolling averages)
- Batch prediction for upcoming race runners

Models predict normalized times (relative to track/distance medians) to handle
different track configurations.

Example:
    >>> from src.core.predictor import RacePredictor
    >>> predictor = RacePredictor()
    >>> runners_df = get_today_runners()
    >>> history_df = get_historical_data()
    >>> predictions = predictor.predict_batch(runners_df, history_df)
    >>> print(predictions[['NameKey', 'PredictedSplit', 'PredictedPace']])
"""

import os
import pickle
import sqlite3
from typing import Optional
import pandas as pd
import numpy as np
import xgboost as xgb

PIR_MODEL_PATH: str = 'models/pir_xgb_model.pkl'
PACE_MODEL_PATH: str = 'models/pace_xgb_model.pkl'
DB_PATH: str = 'greyhound_racing.db'


class RacePredictor:
    """XGBoost-based predictor for split times and finish times.
    
    Loads pre-trained models and track/distance benchmarks on initialization.
    Uses lag features and rolling averages for time-series prediction.
    
    Attributes:
        pir_model: Loaded PIR (split time) XGBoost model
        pace_model: Loaded Pace (finish time) XGBoost model
        benchmarks (pd.DataFrame): Track/distance median times
    
    Example:
        >>> predictor = RacePredictor()
        >>> # Check if models loaded
        >>> print(predictor.pir_model is not None)
        True
    """

    def __init__(self) -> None:
        """Initialize predictor by loading models and benchmarks.
        
        Attempts to load PIR and Pace models from disk. If models don't exist,
        prediction will fall back to NaN (no prediction).
        
        Also loads track/distance benchmarks from historical data for normalization.
        """
        self.pir_model = None
        self.pace_model = None
        self.benchmarks = None
        self._load_models()
        self._load_benchmarks()

    def _load_models(self) -> None:
        """Load pre-trained XGBoost models from disk.
        
        Attempts to load PIR (split time) and Pace (finish time) models.
        Prints status messages. Gracefully handles missing files.
        """
        # Load PIR Model
        if os.path.exists(PIR_MODEL_PATH):
            try:
                with open(PIR_MODEL_PATH, 'rb') as f:
                    self.pir_model = pickle.load(f)
                print(f"[RacePredictor] Loaded PIR XGBoost model")
            except Exception as e:
                print(f"[RacePredictor] Error loading PIR model: {e}")
        
        # Load Pace Model
        if os.path.exists(PACE_MODEL_PATH):
            try:
                with open(PACE_MODEL_PATH, 'rb') as f:
                    self.pace_model = pickle.load(f)
                print(f"[RacePredictor] Loaded Pace XGBoost model")
            except Exception as e:
                print(f"[RacePredictor] Error loading Pace model: {e}")

    def _load_benchmarks(self) -> None:
        """Load track/distance median times from historical database.
        
        Calculates median split times and finish times for each track/distance
        combination. Used for normalizing predictions across different tracks.
        
        Filters invalid times (splits <0 or >30, finish times <15 or >50).
        """
        """Loads static Track/Distance medians for both Split and FinishTime."""
        try:
            conn = sqlite3.connect(DB_PATH)
            # Fetch raw data to calc medians in pandas
            query_raw = """
            SELECT 
                t.TrackName, 
                r.Distance, 
                ge.Split,
                ge.FinishTime
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE ge.Position NOT IN ('DNF', 'SCR', '')
            """
            df = pd.read_sql_query(query_raw, conn)
            conn.close()
            
            # Filter valid ranges
            split_df = df[(df['Split'] > 0) & (df['Split'] < 30)]
            pace_df = df[(df['FinishTime'] > 15) & (df['FinishTime'] < 50)]
            
            # Calc Medians
            split_bench = split_df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
            split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
            
            pace_bench = pace_df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
            pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
            
            # Merge
            self.benchmarks = split_bench.merge(pace_bench, on=['TrackName', 'Distance'], how='outer')
            print(f"[RacePredictor] Loaded benchmarks for {len(self.benchmarks)} Track/Distance pairs")
            
        except Exception as e:
            print(f"[RacePredictor] Error loading benchmarks: {e}")
            self.benchmarks = None

    def predict_batch(
        self,
        runners_df: pd.DataFrame,
        full_history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict split and pace times for batch of runners.
        
        Calculates features from historical data and generates predictions using
        loaded XGBoost models. Returns dataframe with predicted times.
        
        Args:
            runners_df: Today's runners with columns: NameKey, TrackName, Distance, 
                       Box, MeetingDate
            full_history_df: Historical race data for feature engineering with columns:
                            NameKey, TrackName, Distance, Split, FinishTime, MeetingDate
        
        Returns:
            DataFrame with columns: NameKey, PredictedSplit, PredictedPace, DaysSince
        
        Example:
            >>> runners = pd.DataFrame({'NameKey': ['DOG_A'], 'Distance': [500], ...})
            >>> history = get_historical_data()
            >>> predictions = predictor.predict_batch(runners, history)
            >>> print(predictions['PredictedSplit'].iloc[0])
            17.52
        """
        """
        Predicts Split and Pace for a batch of runners.
        Returns: runners_df with 'PredictedSplit' and 'PredictedPace'.
        """
        if self.benchmarks is None:
            return self._fallback_prediction(runners_df, full_history_df)

        # Merge Benchmarks to History
        hist_df = full_history_df.merge(self.benchmarks, on=['TrackName', 'Distance'], how='left')
        
        # Calculate Norms
        hist_df['NormSplit'] = hist_df['Split'] - hist_df['TrackDistMedianSplit']
        hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['TrackDistMedianPace']
        
        # Sort
        hist_df = hist_df.sort_values(['NameKey', 'MeetingDate'])
        
        # Feature Extraction Loop
        grouped = hist_df.groupby('NameKey')
        features_dict = {}
        
        for name, group in grouped:
            if len(group) < 3: continue
            
            # SPLIT FEATURES
            ns = group['NormSplit'].dropna()
            if len(ns) >= 3:
                s_lag1 = ns.iloc[-1]
                s_lag2 = ns.iloc[-2] if len(ns) >= 2 else np.nan
                s_lag3 = ns.iloc[-3] if len(ns) >= 3 else np.nan
                s_roll3 = ns.rolling(window=3, min_periods=3).mean().iloc[-1]
                s_roll5 = ns.rolling(window=5, min_periods=5).mean().iloc[-1]
            else:
                s_lag1 = s_lag2 = s_lag3 = s_roll3 = s_roll5 = np.nan

            # PACE FEATURES
            nt = group['NormTime'].dropna()
            if len(nt) >= 3:
                p_lag1 = nt.iloc[-1]
                p_lag2 = nt.iloc[-2] if len(nt) >= 2 else np.nan
                p_lag3 = nt.iloc[-3] if len(nt) >= 3 else np.nan
                p_roll3 = nt.rolling(window=3, min_periods=3).mean().iloc[-1]
                p_roll5 = nt.rolling(window=5, min_periods=5).mean().iloc[-1]
            else:
                p_lag1 = p_lag2 = p_lag3 = p_roll3 = p_roll5 = np.nan
            
            last_date = group['MeetingDate'].max()
            
            features_dict[name] = {
                's_Lag1': s_lag1, 's_Lag2': s_lag2, 's_Lag3': s_lag3, 's_Roll3': s_roll3, 's_Roll5': s_roll5,
                'p_Lag1': p_lag1, 'p_Lag2': p_lag2, 'p_Lag3': p_lag3, 'p_Roll3': p_roll3, 'p_Roll5': p_roll5,
                'LastRaceDate': last_date
            }
            
        features_df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index()
        features_df.rename(columns={'index': 'NameKey'}, inplace=True)
        
        # Merge to Today
        runners_df['MeetingDate'] = pd.to_datetime(runners_df['MeetingDate'])
        model_input = runners_df.merge(features_df, on='NameKey', how='left')
        model_input['DaysSince'] = (model_input['MeetingDate'] - model_input['LastRaceDate']).dt.days
        
        # Merge Today Benchmark
        model_input = model_input.merge(self.benchmarks, on=['TrackName', 'Distance'], how='left')
        
        # Type conversions
        model_input['Box'] = pd.to_numeric(model_input['Box'], errors='coerce').fillna(0)
        model_input['Distance'] = pd.to_numeric(model_input['Distance'], errors='coerce').fillna(0)

        # PREDICT SPLIT
        if self.pir_model:
            X_split = model_input[['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
            X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance'] # Rename features to match model
            
            # Predict only valid rows
            # XGBoost handles NaNs, but if all features are NaN (new dog), result is meaningless.
            # We trust XGBoost to handle partial history.
            try:
                model_input['PredictedNormSplit'] = self.pir_model.predict(X_split)
                model_input['PredictedSplit'] = model_input['PredictedNormSplit'] + model_input['TrackDistMedianSplit']
            except:
                model_input['PredictedSplit'] = np.nan
        else:
            model_input['PredictedSplit'] = np.nan

        # PREDICT PACE
        if self.pace_model:
            X_pace = model_input[['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']].copy()
            X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance'] 
            
            try:
                model_input['PredictedNormPace'] = self.pace_model.predict(X_pace)
                model_input['PredictedPace'] = model_input['PredictedNormPace'] + model_input['TrackDistMedianPace']
            except:
                model_input['PredictedPace'] = np.nan
        else:
            model_input['PredictedPace'] = np.nan
            
        return model_input[['NameKey', 'PredictedSplit', 'PredictedPace', 'DaysSince']]

    def _fallback_prediction(
        self,
        runners_df: pd.DataFrame,
        full_history_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Fallback prediction when benchmarks unavailable.
        
        Returns NaN for all predictions when models or benchmarks cannot be loaded.
        
        Args:
            runners_df: Today's runners dataframe
            full_history_df: Historical data (unused in fallback)
        
        Returns:
            DataFrame with NaN predictions and DaysSince=999
        """
        # Fallback averages
        res = runners_df.copy()
        res['PredictedSplit'] = np.nan
        res['PredictedPace'] = np.nan
        res['DaysSince'] = 999
        return res[['NameKey', 'PredictedSplit', 'PredictedPace', 'DaysSince']]

