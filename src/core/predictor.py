
import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
DB_PATH = 'greyhound_racing.db'

class RacePredictor:
    def __init__(self):
        self.pir_model = None
        self.pace_model = None
        self.benchmarks = None
        self._load_models()
        self._load_benchmarks()

    def _load_models(self):
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

    def _load_benchmarks(self):
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

    def predict_batch(self, runners_df, full_history_df):
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

    def _fallback_prediction(self, runners_df, full_history_df):
        # Fallback averages
        res = runners_df.copy()
        res['PredictedSplit'] = np.nan
        res['PredictedPace'] = np.nan
        res['DaysSince'] = 999
        return res[['NameKey', 'PredictedSplit', 'PredictedPace', 'DaysSince']]

