"""
Greyhound Racing ML Model
Predicts race winners using XGBoost based on historical performance data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)
import pickle
import json
import sys
import joblib
import os


class GreyhoundMLModel:
    def __init__(self, db_path='greyhound_racing.db'):
        self.db_path = db_path
        # print(f"[DEBUG] GreyhoundMLModel initialized with DB: {os.path.abspath(self.db_path)}")
        self.model = None
        self.track_encoder = LabelEncoder()
        self.feature_columns = []
        self.model_metrics = {}
        
        # Probability calibration parameters (Platt scaling)
        self.calibration_a = None  # Scale parameter
        self.calibration_b = None  # Bias parameter
        self.use_calibration = False

        # Track tier definitions for weighting GM_OT_ADJ
        self.METRO_TRACKS = {
            'Wentworth Park', 'Albion Park', 'Angle Park', 'Hobart',
            'Launceston', 'Sandown Park', 'The Meadows', 'Cannington'
        }

        self.PROVINCIAL_TRACKS = {
            'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli',
            'Dapto', 'Maitland', 'Goulburn', 'Ipswich', 'Q Straight',
            'Q1 Lakeside', 'Q2 Parklands', 'Gawler', 'Devonport', 'Ballarat',
            'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'
        }

        # Track tier weights: Metro=1.0, Provincial=0.7, Country=0.3
        self.TRACK_WEIGHTS = {'metro': 1.0, 'provincial': 0.7, 'country': 0.3}

    def get_track_tier_weight(self, track_name):
        """
        Get tier weight for a track
        Returns: 1.0 for metro, 0.7 for provincial, 0.3 for country
        """
        if track_name in self.METRO_TRACKS:
            return self.TRACK_WEIGHTS['metro']
        elif track_name in self.PROVINCIAL_TRACKS:
            return self.TRACK_WEIGHTS['provincial']
        else:
            return self.TRACK_WEIGHTS['country']

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def extract_training_data(self, start_date='2023-01-01', end_date='2025-06-30'):
        """
        Extract training data for Regression Model (NormTime Prediction)
        Features: DogNormTimeAvg, Box, Distance
        Target: NormTime (FinishTime - MedianTime)
        """
        print(f"Extracting training data from {start_date} to {end_date}...")
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Load Benchmarks
        benchmarks_query = "SELECT TrackName, Distance, MedianTime FROM Benchmarks"
        try:
             bench_df = pd.read_sql_query(benchmarks_query, conn)
             benchmarks_dict = {}
             for _, row in bench_df.iterrows():
                 benchmarks_dict[(row['TrackName'], row['Distance'])] = row['MedianTime']
        except:
             print("[WARN] Could not load benchmarks.")
             benchmarks_dict = {}

        # 2. Extract Raw History
        query = """
        SELECT
            ge.EntryID,
            ge.GreyhoundID,
            ge.Box,
            ge.FinishTime,
            r.Distance,
            t.TrackName,
            rm.MeetingDate,
            ge.StartingPrice
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= ? AND rm.MeetingDate <= ?
          AND ge.FinishTime IS NOT NULL
          AND ge.Position NOT IN ('DNF', 'SCR', '')
        ORDER BY ge.GreyhoundID, rm.MeetingDate
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            if len(df) == 0:
                return None
                
            # Calculate NormTime and DogNormTimeAvg in Python (easier than SQL window for benchmarks)
            # We need to compute DogNormTimeAvg based on PRIOR races
            
            # First, compute NormTime for every row
            def get_norm_time(row):
                median = benchmarks_dict.get((row['TrackName'], row['Distance']))
                if median:
                    return float(row['FinishTime']) - median
                return None

            df['NormTime'] = df.apply(get_norm_time, axis=1)
            df = df.dropna(subset=['NormTime'])
            
            # Now compute sliding window average of NormTime partitioned by Dog
            # Shift 1 to ensure we only use PAST data for the feature
            df['DogNormTimeAvg'] = df.groupby('GreyhoundID')['NormTime'].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
            
            # Fill NaNs (first race for a dog) with 0.0 (Median performance)
            df['DogNormTimeAvg'] = df['DogNormTimeAvg'].fillna(0.0)
            
            # Select final columns
            final_df = df[['DogNormTimeAvg', 'Box', 'Distance', 'NormTime', 'MeetingDate']].copy()
            final_df = final_df.rename(columns={'NormTime': 'Label'}) # Label is the target
            
            print(f"Extracted {len(final_df)} training samples.")
            return final_df

        except Exception as e:
            print(f"Error extracting data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_features(self, df):
        """
        Prepare features for model training (Regression)
        """
        # Features are already extracted by extract_training_data
        self.feature_columns = ['DogNormTimeAvg', 'Box', 'Distance']
        
        # Drop rows where any of the feature columns or the label are NaN
        df_clean = df.dropna(subset=self.feature_columns + ['Label']).copy()
        
        X = df_clean[self.feature_columns]
        y = df_clean['Label']
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: Mean={y.mean():.4f}, Std={y.std():.4f}")

        return X, y, df_clean

    def _extract_greyhound_features(self, conn, greyhound_id, current_date, current_track, current_distance, box, weight, track_id=None, prize_money=None, box_win_rates=None, benchmarks_dict=None, race_id=None):
        """
        Extract features for Regression Model (DogNormTimeAvg, Box, Distance)
        Includes caching mechanism in DogFeatureCache table.
        """
        import sys
        
        # 0. Check Cache First
        cursor = conn.cursor()
        if race_id:
            try:
                cursor.execute("SELECT FeaturesJSON FROM DogFeatureCache WHERE DogID=? AND RaceID=?", (greyhound_id, race_id))
                cached = cursor.fetchone()
                if cached and cached[0]:
                    print(f"[DEBUG] CACHE HIT: Dog {greyhound_id}, Race {race_id}")
                    return json.loads(cached[0])
            except Exception as e:
                # print(f"[WARN] Cache read failed: {e}")
                pass

        print(f"[DEBUG] CACHE MISS: Dog {greyhound_id} (Extracting features logic running...)")
        sys.stdout.flush()

        # Get last 5 races (to compute rolling avg)
        cursor.execute("""
            SELECT
                t.TrackName,
                r.Distance,
                ge.FinishTime
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE ge.GreyhoundID = ?
              AND rm.MeetingDate < ?
              AND ge.Position NOT IN ('DNF', 'SCR', '')
              AND ge.FinishTime IS NOT NULL
            ORDER BY rm.MeetingDate DESC
            LIMIT 5
        """, (greyhound_id, current_date))

        history = cursor.fetchall()
        print(f"[DEBUG] Dog {greyhound_id} history count: {len(history)} (Date < {current_date})")
        sys.stdout.flush()
        
        # Calculate NormTime for each history item
        norm_times = []
        if benchmarks_dict:
            for h_track, h_dist, h_time in history:
                try:
                    h_time = float(h_time)
                    h_dist = int(h_dist)
                    median = benchmarks_dict.get((h_track, h_dist))
                    if median:
                        norm_times.append(h_time - median)
                except (ValueError, TypeError):
                    continue

        # Need at least 1 history point to calculate average
        if not norm_times:
            if len(history) > 0:
                 # Debug why we have history but no norm_times (likely missing benchmarks)
                 print(f"[DEBUG] {greyhound_id}: Has {len(history)} races but 0 norm_times. (Benchmarks missing?)")
                 try:
                     first_hist = history[0]
                     print(f"       Ex: {first_hist[0]} {first_hist[1]}m")
                 except: pass
                 
            # Cache the 'None' result (no history) to avoid re-querying
            if race_id:
                try:
                    # print(f"[DEBUG] CACHE WRITE (None): Dog {greyhound_id}, Race {race_id}")
                    cursor.execute("""
                        INSERT OR REPLACE INTO DogFeatureCache (DogID, RaceID, FeaturesJSON)
                        VALUES (?, ?, ?)
                    """, (greyhound_id, race_id, 'null'))
                    conn.commit()
                except Exception as e:
                    pass

            return None

        # Calculate DogNormTimeAvg (Shift 1, which IS this current set relative to today, Rolling 3)
        # We just take the average of the last 1, 2, or 3 valid norm_times
        valid_norms = norm_times[:3] # Take most recent 3
        dog_norm_time_avg = sum(valid_norms) / len(valid_norms)

        features = {}
        features['DogNormTimeAvg'] = dog_norm_time_avg
        
        # Box
        try:
            features['Box'] = int(box) if box else 0
        except:
            features['Box'] = 0
            
        # Distance
        try:
            features['Distance'] = float(current_distance)
        except:
            features['Distance'] = 0.0

        # Save to Cache
        if race_id:
            try:
                # print(f"[DEBUG] CACHE WRITE: Dog {greyhound_id}, Race {race_id}")
                json_str = json.dumps(features)
                cursor.execute("""
                    INSERT OR REPLACE INTO DogFeatureCache (DogID, RaceID, FeaturesJSON)
                    VALUES (?, ?, ?)
                """, (greyhound_id, race_id, json_str))
                conn.commit() # Force commit immediately to ensure persistence
            except Exception as e:
                print(f"[WARN] Cache write failed: {e}")

        return features

    def predict_race_winners_v2(self, race_date, confidence_threshold=0.1):
        """
        Predict candidates for LAY Strategy using Regression Model
        Threshold = Margin in seconds (default 0.1)
        """
        print(f"\\n[DEBUG] predict_race_winners (Regression) called for: {race_date}, Margin > {confidence_threshold}s")
        conn = self.get_connection()
        
        # 1. Load Benchmarks for NormTime calculation
        benchmarks_query = "SELECT TrackName, Distance, MedianTime FROM Benchmarks"
        bench_df = pd.read_sql_query(benchmarks_query, conn)
        benchmarks_dict = {}
        for _, row in bench_df.iterrows():
            benchmarks_dict[(row['TrackName'], row['Distance'])] = row['MedianTime']
            
        print(f"[DEBUG] Loaded {len(benchmarks_dict)} benchmarks.")
        if len(benchmarks_dict) < 5:
             print(f"[DEBUG] Sample benchmarks: {list(benchmarks_dict.items())}")

        # 2. Get today's entries
        query = """
        SELECT DISTINCT
            ge.EntryID,
            ge.RaceID,
            g.GreyhoundName,
            g.GreyhoundID,
            rm.MeetingDate,
            t.TrackName as CurrentTrack,
            r.RaceNumber,
            r.RaceTime,
            r.Distance,
            ge.Box,
            ge.StartingPrice
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
        ORDER BY r.RaceNumber, ge.Box
        """
        df = pd.read_sql_query(query, conn, params=(race_date,))
        print(f"[DEBUG] Found {len(df)} entries for date {race_date}")

        if len(df) == 0:
            return pd.DataFrame()

        # 3. Extract Features
        features_list = []
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"[DEBUG] Processing entry {idx}/{len(df)}...")

            features = self._extract_greyhound_features(
                conn,
                int(row['GreyhoundID']),
                row['MeetingDate'],
                row['CurrentTrack'],
                row['Distance'],
                row['Box'],
                None, # Weight not used
                None, # TrackID not used
                None, # Prize not used
                None, # Box Stats not used
                benchmarks_dict=benchmarks_dict,
                race_id=int(row['RaceID']) if pd.notna(row['RaceID']) else None  # Pass RaceID for caching
            )

            if features is not None:
                features['EntryID'] = row['EntryID']
                features['RaceID'] = row['RaceID']
                features['GreyhoundName'] = row['GreyhoundName']
                features['RaceNumber'] = row['RaceNumber']
                features['CurrentTrack'] = row['CurrentTrack']
                features['RaceTime'] = row['RaceTime']
                features['StartingPrice'] = row['StartingPrice']
                features_list.append(features)

        # Commit cached features and close connection
        try:
            conn.commit()
            conn.close()
        except: pass

        if not features_list:
            print("[DEBUG] No features extracted (lack of history?)")
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        print(f"[DEBUG] Extracted features for {len(features_df)} entries")

        # 4. Predict NormTime
        X = features_df[self.feature_columns]
        # Handle raw model prediction (some versions might need .predict, others wrap it)
        try:
             # Check if model is raw XGBoost or sklearn wrapper
             if hasattr(self.model, 'predict'):
                 predictions = self.model.predict(X)
             else:
                 # Raw booster requires DMatrix? Not usually if loaded via pickle as object
                 predictions = self.model.predict(X) 
        except Exception as e:
             # Fallback for DMatrix if raw booster
             try:
                 dtest = xgb.DMatrix(X)
                 predictions = self.model.predict(dtest)
             except:
                 print(f"[ERROR] Prediction failed: {e}")
                 return pd.DataFrame()

        features_df['PredNormTime'] = predictions

        # 5. Rank and Calculate Margins per Race
        features_df['PredRank'] = features_df.groupby('RaceID')['PredNormTime'].rank(method='min')
        
        # Get Race Winners (Rank 1)
        rank1s = features_df[features_df['PredRank'] == 1].copy()
        
        # Get 2nd Place Predictions to calc margin
        rank2s = features_df[features_df['PredRank'] == 2][['RaceID', 'PredNormTime']].copy()
        rank2s.columns = ['RaceID', 'Time2nd']
        
        candidates = rank1s.merge(rank2s, on='RaceID', how='left')
        candidates['Margin'] = candidates['Time2nd'] - candidates['PredNormTime']
        
        # 6. Apply Filters (Regression Lay Strategy)
        # Margin > 0.1s (Dominant Winner)
        candidates = candidates[candidates['Margin'] > confidence_threshold]
        
        # Odds Filter: Odds < 2.25 (False Favorites) - from full_walkforward_lay.py
        def safe_convert_odds(x):
            if pd.notna(x) and x not in [0, '0', 'None', None]:
                try:
                    return float(x)
                except:
                    return 100.0
            return 100.0
            
        candidates['Odds'] = candidates['StartingPrice'].apply(safe_convert_odds)
        candidates = candidates[candidates['Odds'] < 2.25]
        # Also ensure odds are valid (>= 1.0)
        candidates = candidates[candidates['Odds'] >= 1.0]

        print(f"[DEBUG] Found {len(candidates)} Lay Candidates (Margin > {confidence_threshold}s, Odds < 2.25)")
        
        # Rename for compatibility
        candidates = candidates.rename(columns={'PredNormTime': 'WinProbability'}) # Hack to pass "score" column check
        
        return candidates[['GreyhoundName', 'RaceNumber', 'CurrentTrack', 'RaceTime', 'Box', 'StartingPrice', 'Margin', 'Odds']]

    def prepare_features(self, df):
        """
        Prepare features for model training
        Uses 9 base features (8 original + new historical finish pace metric)
        """
        print("Preparing features...")

        # Define feature columns (Matched to scripts/train_model.py and _extract_greyhound_features)
        self.feature_columns = ['DogNormTimeAvg', 'Box', 'Distance']

        X = df[self.feature_columns]
        y = df['Label']

        print(f"Features shape: {X.shape}")
        print(f"Target distribution: Winners={y.sum()}, Non-winners={len(y)-y.sum()}")

        return X, y, df

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost Regression model
        """
        print("\nTraining XGBoost Regression model...")

        # XGBoost parameters (Regression - Absolute Error)
        params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'max_depth': 4, # Slightly constrained to prevent overfitting on noise
            'learning_rate': 0.05,
            'n_estimators': 300,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Train model
        self.model = xgb.XGBRegressor(**params)

        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        print("Model training complete!")

        return self.model

    def calibrate_predictions(self, X_val, y_val):
        """
        Calibrate model predictions using Platt scaling on validation set.
        Learns parameters a and b to map raw predictions to actual probabilities.
        
        Problem: Model predicts 70% but actual is only 23.7%
        Solution: Learn scaling function f(x) = 1 / (1 + exp(a*x + b))
        """
        print("\nCalibrating predictions using Platt scaling...")
        
        # Get raw predictions from test set
        raw_probs = self.model.predict_proba(X_val)[:, 1]
        
        # Fit Platt scaling parameters using logistic regression
        # This finds optimal a and b to minimize log loss
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        raw_probs_reshaped = raw_probs.reshape(-1, 1)
        
        # Fit logistic regression to learn calibration parameters
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(raw_probs_reshaped, y_val)
        
        # Extract calibration parameters
        self.calibration_b = calibrator.intercept_[0]
        self.calibration_a = calibrator.coef_[0][0]
        self.use_calibration = True
        
        print(f"Calibration parameters learned:")
        print(f"  a (scale): {self.calibration_a:.6f}")
        print(f"  b (bias):  {self.calibration_b:.6f}")
        
        # Verify calibration on validation set
        calibrated_probs = 1.0 / (1.0 + np.exp(-(self.calibration_a * raw_probs + self.calibration_b)))
        
        # Calculate calibration metrics
        original_logloss = log_loss(y_val, raw_probs)
        calibrated_logloss = log_loss(y_val, np.clip(calibrated_probs, 1e-15, 1-1e-15))
        
        print(f"\nCalibration improvement:")
        print(f"  Original log loss:   {original_logloss:.6f}")
        print(f"  Calibrated log loss: {calibrated_logloss:.6f}")
        print(f"  Improvement:         {(original_logloss - calibrated_logloss):.6f}")
        
        return calibrator

    def apply_calibration(self, raw_probs):
        """
        Apply learned calibration to raw predictions.
        Uses Platt scaling: f(x) = 1 / (1 + exp(a*x + b))
        """
        if not self.use_calibration or self.calibration_a is None:
            return raw_probs
        
        calibrated = 1.0 / (1.0 + np.exp(-(self.calibration_a * raw_probs + self.calibration_b)))
        return np.clip(calibrated, 0, 1)

    def evaluate_model(self, X_train, y_train, X_test, y_test, test_df):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        print(f"\nBasic Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Log Loss:  {logloss:.4f}")

        # Baseline comparisons
        baseline_random = 1/8  # Assuming 8 dogs per race on average
        print(f"\nBaseline Comparison:")
        print(f"  Random guess accuracy: {baseline_random:.4f}")
        print(f"  Model improvement: {(accuracy/baseline_random - 1)*100:.1f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")

        # High confidence predictions (80% threshold)
        high_conf_mask = y_pred_proba >= 0.8
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = y_test[high_conf_mask].mean()
            print(f"\nHigh Confidence Predictions (≥80%):")
            print(f"  Count: {high_conf_mask.sum()}")
            print(f"  Accuracy: {high_conf_accuracy:.4f}")
            print(f"  Expected wins if betting on all: {high_conf_mask.sum() * high_conf_accuracy:.1f}")
        else:
            print(f"\nNo predictions with ≥80% confidence")

        # Profit/Loss Analysis
        test_df_with_pred = test_df.copy()
        test_df_with_pred['PredProba'] = y_pred_proba

        # Store profit/loss stats for GUI
        profit_loss_stats = {}

        high_conf_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= 0.8]
        if len(high_conf_bets) > 0:
            total_bets = len(high_conf_bets)
            wins = high_conf_bets['IsWinner'].sum()
            losses = total_bets - wins

            # Calculate profit assuming $1 bets at starting price
            profit = 0
            for _, bet in high_conf_bets.iterrows():
                if bet['IsWinner'] == 1:
                    # Win: get back stake + winnings
                    try:
                        odds = float(bet['StartingPrice']) if pd.notna(bet['StartingPrice']) else 0
                        profit += (odds - 1)
                    except (ValueError, TypeError):
                        pass
                else:
                    # Loss: lose stake
                    profit -= 1

            print(f"\nProfit/Loss Analysis (≥80% confidence, $1 bets):")
            print(f"  Total bets: {total_bets}")
            print(f"  Wins: {wins}")
            print(f"  Losses: {losses}")
            print(f"  Strike rate: {wins/total_bets*100:.1f}%")
            print(f"  Total profit/loss: ${profit:.2f}")
            print(f"  ROI: {profit/total_bets*100:.1f}%")

            # Store for GUI
            profit_loss_stats['basic_80'] = {
                'total_bets': int(total_bets),
                'wins': int(wins),
                'losses': int(losses),
                'strike_rate': float(wins/total_bets),
                'profit': float(profit),
                'roi': float(profit/total_bets)
            }

        # VALUE BETTING ANALYSIS - Copied exactly from test_final_config.py
        def safe_convert_odds(x):
            if pd.notna(x) and x not in [0, '0', 'None', None]:
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 100
            return 100

        # Test both 80% and 85% confidence (same as test_final_config.py)
        for conf_threshold in [0.80, 0.85]:
            print("\n" + "="*80)
            print(f"VALUE BETTING ANALYSIS - {int(conf_threshold*100)}% CONFIDENCE")
            print("="*80)

            value_bets = test_df_with_pred[test_df_with_pred['PredProba'] >= conf_threshold].copy()

            if len(value_bets) == 0:
                print(f"No bets meet {int(conf_threshold*100)}% confidence threshold")
                continue

            value_bets['ImpliedProb'] = 1 / value_bets['StartingPrice'].apply(safe_convert_odds)
            value_bets = value_bets[value_bets['PredProba'] > value_bets['ImpliedProb']]

            if len(value_bets) == 0:
                print("No value bets found (all model probs <= implied probs)")
                continue

            # Add odds column for bracket analysis
            value_bets['Odds'] = value_bets['StartingPrice'].apply(safe_convert_odds)

            # FILTER: Remove bets under $1.50 odds (consistently losing)
            value_bets_filtered = value_bets[value_bets['Odds'] >= 1.50].copy()

            print(f"\nBEFORE ODDS FILTER (>=1.50):")
            print(f"  Total bets: {len(value_bets)}")

            # Overall stats BEFORE filter
            value_wins_before = value_bets['IsWinner'].sum()
            value_profit_before = 0
            for _, bet in value_bets.iterrows():
                if bet['IsWinner'] == 1:
                    value_profit_before += (bet['Odds'] - 1)
                else:
                    value_profit_before -= 1

            print(f"  Wins: {value_wins_before}")
            print(f"  Strike rate: {value_wins_before/len(value_bets)*100:.1f}%")
            print(f"  Profit/Loss: ${value_profit_before:.2f}")
            print(f"  ROI: {value_profit_before/len(value_bets)*100:.1f}%")

            # Overall stats AFTER filter
            value_wins = value_bets_filtered['IsWinner'].sum()
            value_profit = 0
            for _, bet in value_bets_filtered.iterrows():
                if bet['IsWinner'] == 1:
                    value_profit += (bet['Odds'] - 1)
                else:
                    value_profit -= 1

            print(f"\nAFTER ODDS FILTER (>=1.50):")
            print(f"  Total value bets: {len(value_bets_filtered)}")
            print(f"  Wins: {value_wins}")
            print(f"  Losses: {len(value_bets_filtered) - value_wins}")
            print(f"  Strike rate: {value_wins/len(value_bets_filtered)*100:.1f}%")
            print(f"  Profit/Loss: ${value_profit:.2f}")
            print(f"  ROI: {value_profit/len(value_bets_filtered)*100:.1f}%")
            print(f"  Avg odds: {value_bets_filtered['Odds'].mean():.2f}")

            # Store for GUI
            conf_key = f'value_{int(conf_threshold*100)}'
            profit_loss_stats[conf_key] = {
                'before_filter': {
                    'total_bets': int(len(value_bets)),
                    'wins': int(value_wins_before),
                    'losses': int(len(value_bets) - value_wins_before),
                    'strike_rate': float(value_wins_before/len(value_bets)) if len(value_bets) > 0 else 0,
                    'profit': float(value_profit_before),
                    'roi': float(value_profit_before/len(value_bets)) if len(value_bets) > 0 else 0
                },
                'after_filter': {
                    'total_bets': int(len(value_bets_filtered)),
                    'wins': int(value_wins),
                    'losses': int(len(value_bets_filtered) - value_wins),
                    'strike_rate': float(value_wins/len(value_bets_filtered)) if len(value_bets_filtered) > 0 else 0,
                    'profit': float(value_profit),
                    'roi': float(value_profit/len(value_bets_filtered)) if len(value_bets_filtered) > 0 else 0,
                    'avg_odds': float(value_bets_filtered['Odds'].mean()) if len(value_bets_filtered) > 0 else 0
                }
            }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        print(f"\nALL Feature Importances:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Store metrics
        self.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'baseline_random': baseline_random,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'high_conf_count': int(high_conf_mask.sum()),
            'high_conf_accuracy': float(high_conf_accuracy) if high_conf_mask.sum() > 0 else 0,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'profit_loss': profit_loss_stats
        }

        print("\n" + "="*80)

        return self.model_metrics

    def _extract_greyhound_features_OLD(self, conn, greyhound_id, current_date, current_track, current_distance, box, weight, track_id=None, prize_money=None, box_win_rates=None, benchmarks_dict=None):
        """
        Extract features for Regression Model (DogNormTimeAvg, Box, Distance)
        """
        cursor = conn.cursor()

        # Get last 5 races (to compute rolling avg)
        cursor.execute("""
            SELECT
                t.TrackName,
                r.Distance,
                ge.FinishTime
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE ge.GreyhoundID = ?
              AND rm.MeetingDate < ?
              AND ge.Position NOT IN ('DNF', 'SCR', '')
              AND ge.FinishTime IS NOT NULL
            ORDER BY rm.MeetingDate DESC
            LIMIT 5
        """, (greyhound_id, current_date))

        history = cursor.fetchall()
        
        # Calculate NormTime for each history item
        norm_times = []
        if benchmarks_dict:
            for h_track, h_dist, h_time in history:
                try:
                    h_time = float(h_time)
                    h_dist = int(h_dist)
                    median = benchmarks_dict.get((h_track, h_dist))
                    if median:
                        norm_times.append(h_time - median)
                except (ValueError, TypeError):
                    continue

        # Need at least 1 history point to calculate average
        if not norm_times:
            return None

        # Calculate DogNormTimeAvg (Shift 1, which IS this current set relative to today, Rolling 3)
        # We just take the average of the last 1, 2, or 3 valid norm_times
        valid_norms = norm_times[:3] # Take most recent 3
        dog_norm_time_avg = sum(valid_norms) / len(valid_norms)

        features = {}
        features['DogNormTimeAvg'] = dog_norm_time_avg
        
        # Box
        try:
            features['Box'] = int(box) if box else 0
        except:
            features['Box'] = 0
            
        # Distance
        try:
            features['Distance'] = float(current_distance)
        except:
            features['Distance'] = 0.0

        return features

    def predict_race_winners_OLD(self, race_date, confidence_threshold=0.1):
        """
        Predict candidates for LAY Strategy using Regression Model
        Threshold = Margin in seconds (default 0.1)
        """
        print(f"\n[DEBUG] predict_race_winners (Regression) called for: {race_date}, Margin > {confidence_threshold}s")
        conn = self.get_connection()
        
        # 1. Load Benchmarks for NormTime calculation
        benchmarks_query = "SELECT TrackName, Distance, MedianTime FROM Benchmarks"
        bench_df = pd.read_sql_query(benchmarks_query, conn)
        benchmarks_dict = {}
        for _, row in bench_df.iterrows():
            benchmarks_dict[(row['TrackName'], row['Distance'])] = row['MedianTime']

        # 2. Get today's entries
        query = """
        SELECT DISTINCT
            ge.EntryID,
            ge.RaceID,
            g.GreyhoundName,
            g.GreyhoundID,
            rm.MeetingDate,
            t.TrackName as CurrentTrack,
            r.RaceNumber,
            r.RaceTime,
            r.Distance,
            ge.Box,
            ge.StartingPrice
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
        ORDER BY r.RaceNumber, ge.Box
        """
        df = pd.read_sql_query(query, conn, params=(race_date,))
        print(f"[DEBUG] Found {len(df)} entries for date {race_date}")

        if len(df) == 0:
            return pd.DataFrame()

        # 3. Extract Features
        features_list = []
        for idx, row in df.iterrows():
            features = self._extract_greyhound_features(
                conn,
                row['GreyhoundID'],
                row['MeetingDate'],
                row['CurrentTrack'],
                row['Distance'],
                row['Box'],
                None, # Weight not used
                None, # TrackID not used
                None, # Prize not used
                None, # Box Stats not used
                benchmarks_dict=benchmarks_dict,
                race_id=row['RaceID']  # Pass RaceID for caching
            )

            if features is not None:
                features['EntryID'] = row['EntryID']
                features['RaceID'] = row['RaceID']
                features['GreyhoundName'] = row['GreyhoundName']
                features['RaceNumber'] = row['RaceNumber']
                features['CurrentTrack'] = row['CurrentTrack']
                features['RaceTime'] = row['RaceTime']
                features['StartingPrice'] = row['StartingPrice']
                features_list.append(features)

        # Commit cached features and close connection
        try:
            conn.commit()
            conn.close()
        except: pass

        if not features_list:
            print("[DEBUG] No features extracted (lack of history?)")
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        print(f"[DEBUG] Extracted features for {len(features_df)} entries")

        # 4. Predict NormTime
        X = features_df[self.feature_columns]
        # Handle raw model prediction (some versions might need .predict, others wrap it)
        try:
             # Check if model is raw XGBoost or sklearn wrapper
             if hasattr(self.model, 'predict'):
                 predictions = self.model.predict(X)
             else:
                 # Raw booster requires DMatrix? Not usually if loaded via pickle as object
                 predictions = self.model.predict(X) 
        except Exception as e:
             # Fallback for DMatrix if raw booster
             try:
                 dtest = xgb.DMatrix(X)
                 predictions = self.model.predict(dtest)
             except:
                 print(f"[ERROR] Prediction failed: {e}")
                 return pd.DataFrame()

        features_df['PredNormTime'] = predictions

        # 5. Rank and Calculate Margins per Race
        features_df['PredRank'] = features_df.groupby('RaceID')['PredNormTime'].rank(method='min')
        
        # Get Race Winners (Rank 1)
        rank1s = features_df[features_df['PredRank'] == 1].copy()
        
        # Get 2nd Place Predictions to calc margin
        rank2s = features_df[features_df['PredRank'] == 2][['RaceID', 'PredNormTime']].copy()
        rank2s.columns = ['RaceID', 'Time2nd']
        
        candidates = rank1s.merge(rank2s, on='RaceID', how='left')
        candidates['Margin'] = candidates['Time2nd'] - candidates['PredNormTime']
        
        # 6. Apply Filters (Regression Lay Strategy)
        # Margin > Threshold (default 0.1s -> Dominant Winner)
        candidates = candidates[candidates['Margin'] > confidence_threshold]
        
        # Odds Filter: Odds < 2.50 (False Favorites) - strictly applied here or in script? 
        # User script optimizes max odds. Let's use 2.50 as a safe default based on screenshot.
            
        candidates['Odds'] = candidates['StartingPrice'].apply(safe_convert_odds)
        candidates = candidates[candidates['Odds'] < 2.50]
        # Also ensure odds are valid (>= 1.0)
        candidates = candidates[candidates['Odds'] >= 1.0]

        print(f"[DEBUG] Found {len(candidates)} Lay Candidates (Margin > {confidence_threshold}s, Odds < 2.50)")
        
        # Rename for compatibility
        candidates = candidates.rename(columns={'PredNormTime': 'WinProbability'}) # Hack to pass "score" column check
        # Actually, let's just return what the script expects. The script expects 'WinProbability' column?
        # Yes, predict_lay_strategy prints 'Probability: {:.1f}%'. 
        # We should probably supply 'Margin' as the score? or just PredNormTime?
        # I'll leverage flexible column usage in the calling script.
        
        return candidates[['GreyhoundName', 'RaceNumber', 'CurrentTrack', 'RaceTime', 'Box', 'StartingPrice', 'Margin', 'Odds']]

    def cache_all_upcoming_features(self):
        """
        Pre-calculate and cache features for all future races in the database.
        This speeds up 'Load Tips' significantly.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all future dates from Races table
        cursor.execute("""
            SELECT DISTINCT rm.MeetingDate 
            FROM Races r 
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID 
            WHERE rm.MeetingDate >= date('now') 
            ORDER BY rm.MeetingDate
        """)
        dates = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"\\n[CACHE] Found {len(dates)} upcoming dates to cache features for: {dates}")
        
        for date_str in dates:
            print(f"[CACHE] Processing {date_str}...")
            # We use predict_race_winners_v2 purely for its side-effect of caching features
            # The function now checks cache first, and saves to cache if missing.
            try:
                self.predict_race_winners_v2(date_str)
            except Exception as e:
                print(f"[WARN] Failed to cache for {date_str}: {e}")
                
        return len(dates)

    def predict_upcoming_races(self, race_date, confidence_threshold=0.8):
        """
        Predict winners for upcoming races (races that haven't been run yet)
        Uses UpcomingBettingRaces and UpcomingBettingRunners tables

        Args:
            race_date: Date string in YYYY-MM-DD format
            confidence_threshold: Minimum confidence level (0-1)

        Returns: DataFrame with predictions above confidence threshold
        """
        print(f"\n[DEBUG] predict_upcoming_races called for date: {race_date}, threshold: {confidence_threshold}")
        conn = self.get_connection()

        # Get all upcoming races for the date
        # Use MAX to get the most recent non-null odds when there are duplicates
        query = """
        SELECT
            MIN(ur.UpcomingBettingRunnerID) as EntryID,
            ur.GreyhoundName,
            ubr.MeetingDate,
            ubr.TrackName as CurrentTrack,
            ubr.TrackCode,
            ubr.RaceNumber,
            ubr.RaceTime,
            ubr.Distance,
            ur.BoxNumber as Box,
            MAX(CASE WHEN ur.CurrentOdds IS NOT NULL THEN ur.CurrentOdds ELSE NULL END) as CurrentOdds,
            MAX(ur.TrainerName) as TrainerName,
            MAX(ur.Form) as Form,
            MAX(ur.BestTime) as BestTime,
            MAX(ur.Weight) as Weight
        FROM UpcomingBettingRunners ur
        JOIN UpcomingBettingRaces ubr ON ur.UpcomingBettingRaceID = ubr.UpcomingBettingRaceID
        WHERE ubr.MeetingDate = ?
        GROUP BY ur.GreyhoundName, ubr.TrackName, ubr.RaceNumber, ur.BoxNumber, ubr.MeetingDate, ubr.TrackCode, ubr.RaceTime, ubr.Distance
        ORDER BY ubr.RaceNumber, ur.BoxNumber
        """

        df = pd.read_sql_query(query, conn, params=(race_date,))
        print(f"[DEBUG] Found {len(df)} upcoming entries for date {race_date}")

        # Filter out NZ and Tasmania tracks
        excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                          'Launceston', 'Hobart', 'Devonport']
        df = df[~df['CurrentTrack'].isin(excluded_tracks)]
        # Also filter out tracks with (NZ) or NZ in the name (case insensitive)
        df = df[~df['CurrentTrack'].str.contains('NZ', na=False, case=False)]
        print(f"[DEBUG] After filtering NZ/TAS tracks: {len(df)} entries")

        if len(df) == 0:
            print(f"[DEBUG] No upcoming entries found for date {race_date}")
            conn.close()
            return pd.DataFrame()

        # Try to match track names to historical track IDs
        track_query = "SELECT TrackID, TrackName, TrackKey FROM Tracks"
        tracks_df = pd.read_sql_query(track_query, conn)

        # Create mapping from track name to track ID
        track_mapping = {}
        for _, track_row in tracks_df.iterrows():
            track_mapping[track_row['TrackName']] = track_row['TrackID']
            # Also try matching by track code
            if '_' in str(track_row['TrackKey']):
                track_code = track_row['TrackKey'].split('_')[0]
                track_mapping[track_code] = track_row['TrackID']

        # Add TrackID to dataframe
        df['TrackID'] = df['CurrentTrack'].map(track_mapping)
        df['TrackID'] = df['TrackID'].fillna(df['TrackCode'].map(track_mapping))

        print(f"[DEBUG] Matched {df['TrackID'].notna().sum()} entries to historical tracks")

        # Calculate box win rates using all historical data (before this date)
        box_stats_query = """
            SELECT
                t.TrackID,
                r.Distance,
                ge.Box,
                COUNT(*) as TotalRaces,
                SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE rm.MeetingDate < ?
              AND ge.Position IS NOT NULL
              AND ge.Box IS NOT NULL
            GROUP BY t.TrackID, r.Distance, ge.Box
        """
        box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(race_date,))
        box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']

        # Create lookup dict
        box_win_rates = {}
        for _, box_row in box_stats_df.iterrows():
            key = (box_row['TrackID'], box_row['Distance'], box_row['Box'])
            box_win_rates[key] = box_row['BoxWinRate']

        # Try to find greyhound IDs from historical data
        greyhound_id_query = "SELECT GreyhoundID, GreyhoundName FROM Greyhounds"
        greyhounds_df = pd.read_sql_query(greyhound_id_query, conn)
        greyhound_mapping = dict(zip(greyhounds_df['GreyhoundName'], greyhounds_df['GreyhoundID']))

        # Extract features for each entry
        features_list = []
        for idx, row in df.iterrows():
            greyhound_id = greyhound_mapping.get(row['GreyhoundName'])
            track_id = row['TrackID']

            if greyhound_id is None or pd.isna(track_id):
                # Skip greyhounds we haven't seen before or unknown tracks
                print(f"[DEBUG] Skipping {row['GreyhoundName']} - no historical data")
                continue

            features = self._extract_greyhound_features(
                conn,
                greyhound_id,
                row['MeetingDate'],
                row['CurrentTrack'],
                row['Distance'],
                row['Box'],
                row['Weight'] if pd.notna(row['Weight']) else None,
                int(track_id),
                None,  # PrizeMoney not available for upcoming races
                box_win_rates
            )

            if features is not None:
                features['EntryID'] = row['EntryID']
                features['GreyhoundName'] = row['GreyhoundName']
                features['RaceNumber'] = row['RaceNumber']
                features['RaceTime'] = row['RaceTime']
                features['CurrentTrack'] = row['CurrentTrack']
                features['CurrentOdds'] = row['CurrentOdds']
                features_list.append(features)

        conn.close()

        if not features_list:
            print(f"[DEBUG] No features extracted - all greyhounds are new or unknown tracks")
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        print(f"[DEBUG] Extracted features for {len(features_df)} entries")

        # Prepare features for prediction
        X = features_df[self.feature_columns]

        # Make predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        features_df['WinProbability'] = y_pred_proba

        # Filter by confidence threshold
        high_confidence = features_df[features_df['WinProbability'] >= confidence_threshold].copy()
        print(f"[DEBUG] Found {len(high_confidence)} predictions above {confidence_threshold*100}% confidence")

        if len(high_confidence) == 0:
            return pd.DataFrame()

        # Apply value bet filter if odds are available
        high_confidence['Odds'] = high_confidence['CurrentOdds'].apply(
            lambda x: float(x) if pd.notna(x) and x not in [0, '0', 'None', None] else None
        )

        # Filter for value bets (model probability > implied probability from odds)
        value_bets = high_confidence.copy()
        if 'Odds' in value_bets.columns:
            # Calculate implied probability from odds
            value_bets['ImpliedProb'] = value_bets['Odds'].apply(
                lambda x: 1/x if pd.notna(x) and x > 0 else float('nan')
            )
            # Filter where we have an edge (only compare where both values exist)
            value_bets = value_bets[
                (value_bets['ImpliedProb'].isna()) |  # Keep if no odds available
                ((value_bets['WinProbability'].notna()) &
                 (value_bets['ImpliedProb'].notna()) &
                 (value_bets['WinProbability'] > value_bets['ImpliedProb']))  # Or if we have an edge
            ]
            print(f"[DEBUG] Value bets after edge filter: {len(value_bets)}")

            # Apply odds filter (>= $1.50) only if odds are available
            value_bets_with_odds = value_bets[value_bets['Odds'].notna()]
            if len(value_bets_with_odds) > 0:
                value_bets_with_odds = value_bets_with_odds[value_bets_with_odds['Odds'] >= 1.50]
                value_bets_no_odds = value_bets[value_bets['Odds'].isna()]
                value_bets = pd.concat([value_bets_with_odds, value_bets_no_odds])
                print(f"[DEBUG] Value bets after odds filter (>= $1.50): {len(value_bets)}")

        # Sort by probability
        value_bets = value_bets.sort_values('WinProbability', ascending=False)

        return value_bets[['GreyhoundName', 'RaceNumber', 'CurrentTrack', 'RaceTime',
                          'Odds', 'WinProbability']].rename(columns={'Odds': 'CurrentOdds'})

    def save_model(self, filepath='greyhound_model.pkl'):
        """Save trained model and encoders"""
        model_data = {
            'model': self.model,
            'track_encoder': self.track_encoder,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'calibration_a': self.calibration_a,
            'calibration_b': self.calibration_b,
            'use_calibration': self.use_calibration
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='greyhound_model.pkl'):
        """Load trained model and encoders"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Handle different model versions/structures
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            
            # 1. Legacy Classification Model
            if 'track_encoder' in model_data:
                self.track_encoder = model_data['track_encoder']
                self.feature_columns = model_data['feature_columns']
                self.model_metrics = model_data['model_metrics']
            
            # 2. New Regression Model (Lay Strategy)
            elif 'metrics' in model_data:
                self.model_metrics = model_data['metrics']
                # REQUIRED: Set feature columns for prediction
                self.feature_columns = ['DogNormTimeAvg', 'Box', 'Distance']
                
                # Inject feature importance into metrics for GUI compatibility
                if 'feature_importance' in model_data:
                    self.model_metrics['feature_importance'] = model_data['feature_importance']
                # Benchmarks are handled separately or inside app
            
            # Common optional fields
            self.calibration_a = model_data.get('calibration_a', None)
            self.calibration_b = model_data.get('calibration_b', None)
            self.use_calibration = model_data.get('use_calibration', False)
        else:
             # Assume it's a raw model object (e.g. XGBRegressor/Classifier)
             print("[WARN] Loaded raw model object. Using default checks.")
             self.model = model_data

        # print(f"Model loaded from {filepath}")


def main():
    """Train and evaluate the model"""
    print("="*80)
    print("GREYHOUND RACING ML MODEL TRAINING")
    print("="*80)

    # Initialize model
    model = GreyhoundMLModel()

    # Extract training data (2023-2024)
    print("\nExtracting TRAINING data (2023-2024)...")
    train_df = model.extract_training_data(start_date='2023-01-01', end_date='2024-12-31')

    if train_df is None or len(train_df) == 0:
        print("ERROR: Failed to extract training data")
        return

    # Extract test data (2025 H1)
    print("\nExtracting TEST data (2025 H1)...")
    test_df = model.extract_training_data(start_date='2025-01-01', end_date='2025-06-30')

    if test_df is None or len(test_df) == 0:
        print("WARNING: No test data found for 2025")
        test_df = None

    # Prepare training features
    print("\nPreparing training features...")
    X_train, y_train, train_df_with_features = model.prepare_features(train_df)

    # Prepare test features if available
    if test_df is not None:
        print("Preparing test features...")
        # Need to ensure track encoder is already fitted
        X_test, y_test, test_df_with_features = model.prepare_features(test_df)
    else:
        # Fall back to random split if no 2025 data
        print("No 2025 data available, using 30% of 2023-2024 for testing...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        test_indices = X_test.index
        test_df_with_features = train_df_with_features.loc[test_indices]

    # Train model
    model.train_model(X_train, y_train, X_test, y_test)

    # Evaluate model
    metrics = model.evaluate_model(X_train, y_train, X_test, y_test, test_df_with_features)

    # Save model
    model.save_model()

    print("\nTraining complete! Model saved.")

    # Test prediction on a sample date from test set
    print("\n" + "="*80)
    print("TESTING PREDICTIONS")
    print("="*80)
    test_date = '2025-06-01'
    print(f"\nPredicting winners for {test_date}...")
    predictions = model.predict_race_winners(test_date, confidence_threshold=0.8)

    if len(predictions) > 0:
        print(f"\nFound {len(predictions)} high-confidence bets:")
        print(predictions.to_string(index=False))
    else:
        print("\nNo high-confidence predictions for this date")


if __name__ == '__main__':
    main()
