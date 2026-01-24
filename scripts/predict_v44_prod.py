import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sqlite3
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

class MarketAlphaEngine:
    """
    V44/V45 Production Engine
    - BACK Strategy: V44 Steamer (Threshold 0.35 Flat)
    - LAY Strategy: V45 Drifter (Threshold 0.55 Flat)
    - Historical Data Cached for Fast Predictions
    """
    def __init__(self, db_path='greyhound_racing.db'):
        print("Initializing Market Alpha Engine (V44/V45 Production)...")
        self.fe = FeatureEngineerV41()
        self.db_path = db_path
        
        # Load Models - PRODUCTION (Trained on All Data, with Strict Logic)
        self.model_v41 = joblib.load('models/xgb_v41_final.pkl')
        print("Loading Production Models (V44 / V45)...")
        # BACK Strategy: Production Model (Full 2020-Present training)
        self.model_v44 = joblib.load('models/xgb_v44_production.pkl')
        
        # V45 Drifter Model (Production)
        try:
            self.model_v45 = joblib.load('models/xgb_v45_production.pkl')
            self.has_drifter = True
        except:
            print("[WARN] V45 Drifter Production model not found. Lay signals disabled.")
            self.has_drifter = False
            
        self.features_v41 = self.fe.get_feature_list()
        
        # Features required by V44/V45 Models
        self.v44_features = [
            'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
            'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
            'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50'
        ]
        
        # V45 Features (Uses Drift Lags instead of Steam Lags)
        self.v45_features = [
            'Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff',
            'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate',
            'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50'
        ]
        
        # --- HISTORICAL DATA CACHE ---
        self._history_cache = None
        self._cache_date = None
        self._load_history_cache()
        
    def _load_history_cache(self):
        """Load 365 days of historical data into memory cache."""
        import sqlite3
        print("[CACHE] Loading 365 days historical data...")
        start = datetime.now()
        
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT 
            ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
            ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
            ge.Split, ge.FinishTime, ge.Margin,
            r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
            g.DateWhelped
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE rm.MeetingDate >= '{start_date}'
        AND ge.Price5Min > 0
        AND ge.BSP > 0
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Pre-process: V41 Features + Targets + Rolling Features
        df = self.fe.calculate_features(df)
        
        df = df.sort_values('MeetingDate')
        df['MoveRatio'] = df['Price5Min'] / df['BSP']
        df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
        df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)
        
        # Steam Lags
        df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
        df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(
            lambda x: x.rolling(window=10, min_periods=3).mean()
        ).fillna(0)
        df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
        df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
            lambda x: x.rolling(window=50, min_periods=10).mean()
        ).fillna(0)
        
        # Drift Lags
        df['Prev_Drift'] = df.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
        df['Dog_Rolling_Drift_10'] = df.groupby('GreyhoundID')['Prev_Drift'].transform(
            lambda x: x.rolling(window=10, min_periods=3).mean()
        ).fillna(0)
        df['Trainer_Prev_Drift'] = df.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
        df['Trainer_Rolling_Drift_50'] = df.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
            lambda x: x.rolling(window=50, min_periods=10).mean()
        ).fillna(0)
        
        # V41 Probabilities
        # V41 Probabilities
        for c in self.features_v41:
            if c not in df.columns: df[c] = 0
            
        try:
            df['V41_Prob'] = self.model_v41.predict_proba(df[self.features_v41])[:, 1]
        except:
            dtest = xgb.DMatrix(df[self.features_v41])
            df['V41_Prob'] = self.model_v41.predict(dtest)
        df['V41_Price'] = 1.0 / df['V41_Prob']
        df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
        df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
        
        self._history_cache = df
        self._cache_date = datetime.now().strftime('%Y-%m-%d')
        
        elapsed = (datetime.now() - start).total_seconds()
        print(f"[CACHE] Loaded {len(df)} rows in {elapsed:.1f}s. Cache date: {self._cache_date}")
    
    def refresh_cache(self):
        """Force refresh the historical cache."""
        self._load_history_cache()
        
    def _get_cached_features(self, dog_id, trainer_id):
        """Get pre-computed rolling features from cache for a dog/trainer."""
        if self._history_cache is None:
            return {}

        # Ensure IDs are integers for robust lookup (Input might be float from Pandas with NaNs)
        # Ensure IDs are integers for robust lookup (Input might be float from Pandas with NaNs)
        try:
            if pd.notna(dog_id):
                dog_id = int(float(dog_id)) # Handle '123.0' string or float
            else:
                dog_id = None
        except (ValueError, TypeError):
            pass # Keep as is if conversion fails

        try:
            if pd.notna(trainer_id):
                trainer_id = int(float(trainer_id)) # Handle '123.0' string or float
            else:
                trainer_id = None
        except (ValueError, TypeError):
            pass
            
        # Get latest row for this dog
        dog_df = self._history_cache[self._history_cache['GreyhoundID'] == dog_id]
        if dog_df.empty:
            dog_steam = 0
            dog_drift = 0
        else:
            # ROBUST CACHE: Compute rolling mean from the tail of the history
            # This works even if "Today's" row is missing from cache (it uses the last N races)
            # This is correct because Today's rolling stat depends only on Past races.
            dog_steam = dog_df['Is_Steamer_Hist'].tail(10).mean()
            if len(dog_df) < 3: dog_steam = 0 # min_periods=3
            
            dog_drift = dog_df['Is_Drifter_Hist'].tail(10).mean()
            if len(dog_df) < 3: dog_drift = 0
            
        # Get latest row for this trainer
        trainer_df = self._history_cache[self._history_cache['TrainerID'] == trainer_id]
        if trainer_df.empty:
            trainer_steam = 0
            trainer_drift = 0
        else:
            trainer_steam = trainer_df['Is_Steamer_Hist'].tail(50).mean()
            if len(trainer_df) < 10: trainer_steam = 0 # min_periods=10
            
            trainer_drift = trainer_df['Is_Drifter_Hist'].tail(50).mean()
            if len(trainer_df) < 10: trainer_drift = 0
            

        return {
            'Dog_Rolling_Steam_10': dog_steam,
            'Dog_Rolling_Drift_10': dog_drift,
            'Trainer_Rolling_Steam_50': trainer_steam,
            'Trainer_Rolling_Drift_50': trainer_drift
        }

    def predict(self, df_input, use_cache=True):
        """
        Generate predictions. If use_cache=True, uses pre-computed rolling features.
        """
        # 1. Generate V41 Base Features
        df = self.fe.calculate_features(df_input.copy())
        
        # 2. Get Fair Value Probability (V41)
        for c in self.features_v41:
            if c not in df.columns: df[c] = 0
            
        try:
            df['V41_Prob'] = self.model_v41.predict_proba(df[self.features_v41])[:, 1]
        except:
            dtest = xgb.DMatrix(df[self.features_v41])
            df['V41_Prob'] = self.model_v41.predict(dtest)
            
        df['V41_Price'] = 1.0 / df['V41_Prob']
        
        # 3. Generate Alpha Features
        # FALLBACK: If Price5Min is missing, try LivePrice, Back, CurrentOdds, or even BSP (if testing)
        if 'LivePrice' in df.columns:
            df['Price5Min'] = df['Price5Min'].fillna(df['LivePrice'])
        if 'Back' in df.columns:
            df['Price5Min'] = df['Price5Min'].fillna(df['Back'])
        if 'CurrentOdds' in df.columns:
            df['Price5Min'] = df['Price5Min'].fillna(df['CurrentOdds'])
            
        df['Discrepancy'] = df['Price5Min'] / df['V41_Price']
        df['Price_Diff'] = df['Price5Min'] - df['V41_Price']
        
        # 4. Rolling Features (Cache or Compute)
        if use_cache and self._history_cache is not None:
            # FAST PATH: Use cached rolling features
            df['Dog_Rolling_Steam_10'] = 0.0
            df['Dog_Rolling_Drift_10'] = 0.0
            df['Trainer_Rolling_Steam_50'] = 0.0
            df['Trainer_Rolling_Drift_50'] = 0.0
            
            for idx, row in df.iterrows():
                dog_id = row.get('GreyhoundID')
                trainer_id = row.get('TrainerID')
                cached = self._get_cached_features(dog_id, trainer_id)
                df.at[idx, 'Dog_Rolling_Steam_10'] = cached.get('Dog_Rolling_Steam_10', 0)
                df.at[idx, 'Dog_Rolling_Drift_10'] = cached.get('Dog_Rolling_Drift_10', 0)
                df.at[idx, 'Trainer_Rolling_Steam_50'] = cached.get('Trainer_Rolling_Steam_50', 0)
                df.at[idx, 'Trainer_Rolling_Drift_50'] = cached.get('Trainer_Rolling_Drift_50', 0)
        else:
            # SLOW PATH: Compute from scratch (requires historical data in df_input)
            df = df.sort_values('MeetingDate')
            
            if 'BSP' in df.columns:
                # Defined Hist Targets
                df['MoveRatio'] = df['Price5Min'] / df['BSP']
                df['Is_Steamer_Hist'] = (df['MoveRatio'] > 1.15).astype(int)
                df['Is_Drifter_Hist'] = (df['MoveRatio'] < 0.95).astype(int)
                
                df['Prev_Steam'] = df.groupby('GreyhoundID')['Is_Steamer_Hist'].shift(1)
                df['Dog_Rolling_Steam_10'] = df.groupby('GreyhoundID')['Prev_Steam'].transform(
                    lambda x: x.rolling(window=10, min_periods=3).mean()
                ).fillna(0)
                
                df['Trainer_Prev_Steam'] = df.groupby('TrainerID')['Is_Steamer_Hist'].shift(1)
                df['Trainer_Rolling_Steam_50'] = df.groupby('TrainerID')['Trainer_Prev_Steam'].transform(
                    lambda x: x.rolling(window=50, min_periods=10).mean()
                ).fillna(0)
                
                df['Prev_Drift'] = df.groupby('GreyhoundID')['Is_Drifter_Hist'].shift(1)
                df['Dog_Rolling_Drift_10'] = df.groupby('GreyhoundID')['Prev_Drift'].transform(
                    lambda x: x.rolling(window=10, min_periods=3).mean()
                ).fillna(0)
                
                df['Trainer_Prev_Drift'] = df.groupby('TrainerID')['Is_Drifter_Hist'].shift(1)
                df['Trainer_Rolling_Drift_50'] = df.groupby('TrainerID')['Trainer_Prev_Drift'].transform(
                    lambda x: x.rolling(window=50, min_periods=10).mean()
                ).fillna(0)
            else:
                df['Dog_Rolling_Steam_10'] = 0
                df['Trainer_Rolling_Steam_50'] = 0
                df['Dog_Rolling_Drift_10'] = 0
                df['Trainer_Rolling_Drift_50'] = 0
            
        # 5. Predict V44 Steamer
        for c in self.v44_features:
            if c not in df.columns: df[c] = 0
            
        X_v44 = df[self.v44_features]
        df['Steam_Prob'] = self.model_v44.predict_proba(X_v44)[:, 1]
        
        # 6. Predict V45 Drifter
        df['Drift_Prob'] = 0.0
        if self.has_drifter:
            for c in self.v45_features:
                if c not in df.columns: df[c] = 0
            
            X_v45 = df[self.v45_features]
            df['Drift_Prob'] = self.model_v45.predict_proba(X_v45)[:, 1]

        # 7. Signal Logic
        df['Signal'] = 'PASS'
        
        # BACK logic (V44 Production Model)
        # Flat Threshold: 0.35, Price Cap: $30
        mask_back = (df['Steam_Prob'] >= 0.35) & (df['Price5Min'] < 30.0)
        
        df.loc[mask_back, 'Signal'] = 'BACK'
        df.loc[mask_back, 'Confidence'] = df['Steam_Prob']
        
        # LAY logic (V45 Production Model)
        # Flat Threshold: 0.55, Price Cap: $30
        if self.has_drifter:
            mask_lay = (df['Drift_Prob'] >= 0.55) & (df['Price5Min'] < 30.0)
            
            df.loc[mask_lay, 'Signal'] = 'LAY'
            df.loc[mask_lay, 'Confidence'] = df['Drift_Prob'] 
            
        # 8. Global Exclusions
        if 'TrackName' in df.columns:
            tas_tracks = ['LAUNCESTON', 'HOBART', 'DEVONPORT']
            mask_tas = df['TrackName'].str.upper().apply(lambda x: any(t in str(x).upper() for t in tas_tracks))
            if mask_tas.any():
                df.loc[mask_tas, 'Signal'] = 'PASS'
        
        return df

def simulate_signals():
    pass

if __name__ == "__main__":
    simulate_signals()
