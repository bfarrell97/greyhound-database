"""
COMPREHENSIVE MODEL IMPROVEMENT TESTING
Test all improvements with detailed analysis of impact
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle
import sys
from datetime import datetime

class ModelVariantTester:
    """Test different model configurations and measure impact"""
    
    def __init__(self, db_path='greyhound_racing.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Track tier definitions
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
        
        self.TRACK_WEIGHTS = {'metro': 1.0, 'provincial': 0.7, 'country': 0.3}
        
        # Recency weight configurations
        self.recency_configs = {
            'baseline': [2.0, 1.5, 1.0, 1.0, 1.0],
            'aggressive': [3.0, 1.5, 0.8, 0.5, 0.3],
            'moderate': [2.0, 1.5, 1.2, 0.9, 0.7],
            'balanced': [2.5, 1.5, 1.0, 1.0, 1.0],
        }
    
    def get_track_tier_weight(self, track_name):
        if track_name in self.METRO_TRACKS:
            return self.TRACK_WEIGHTS['metro']
        elif track_name in self.PROVINCIAL_TRACKS:
            return self.TRACK_WEIGHTS['provincial']
        else:
            return self.TRACK_WEIGHTS['country']
    
    def extract_training_data(self, recency_weights, include_new_features=False):
        """Extract training data with specified recency weights and optional new features"""
        print(f"\nExtracting training data (2023-01-01 to 2025-05-31)...")
        
        # Get base races
        query = """
        SELECT DISTINCT
            ge.EntryID, g.GreyhoundID, g.GreyhoundName,
            rm.MeetingDate, t.TrackName, r.Distance,
            ge.Box, ge.Weight, ge.Position, ge.StartingPrice
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= '2023-01-01' AND rm.MeetingDate <= '2025-05-31'
          AND ge.Position IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND t.TrackName NOT IN ('Addington', 'Cambridge')
        ORDER BY rm.MeetingDate DESC
        """
        
        races = pd.read_sql_query(query, self.conn)
        print(f"Loaded {len(races)} race entries")
        
        # Get historical data
        hist_query = """
        SELECT ge.GreyhoundID, rm.MeetingDate,
               ge.Position, ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths,
               t.TrackName, r.Distance, ge.Weight
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.Position IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
        ORDER BY ge.GreyhoundID, rm.MeetingDate DESC
        """
        
        hist_data = pd.read_sql_query(hist_query, self.conn)
        hist_grouped = hist_data.groupby('GreyhoundID')
        
        # Extract features
        features_list = []
        for idx, row in races.iterrows():
            if idx % 100000 == 0:
                print(f"  {idx}/{len(races)}")
                sys.stdout.flush()
            
            if row['GreyhoundID'] not in hist_grouped.groups:
                continue
            
            greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
            greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
            last_5 = greyhound_hist.head(5)
            
            if len(last_5) < 5:
                continue
            
            features = {}
            features['BoxWinRate'] = 0.125
            
            # Recent form
            last_3 = last_5.head(3)
            last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
            features['AvgPositionLast3'] = last_3_positions.mean()
            features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3) if len(last_3) > 0 else 0
            
            # GM_OT_ADJ with recency weighting
            for i, (_, race) in enumerate(last_5.iterrows(), 1):
                g_ot_adj = (race['FinishTimeBenchmarkLengths'] or 0)
                m_ot_adj = (race['MeetingAvgBenchmarkLengths'] or 0)
                
                track_weight = self.get_track_tier_weight(race['TrackName'])
                recency_weight = recency_weights[i - 1]
                features[f'GM_OT_ADJ_{i}'] = (g_ot_adj + m_ot_adj) * track_weight * recency_weight
            
            # NEW FEATURES
            if include_new_features:
                # Weight class
                weight = row['Weight']
                if pd.notna(weight):
                    if weight < 30:
                        features['WeightClass'] = 0  # Light
                    elif weight <= 33:
                        features['WeightClass'] = 1  # Medium
                    else:
                        features['WeightClass'] = 2  # Heavy
                else:
                    features['WeightClass'] = 1
                
                # Distance category (normalized)
                distance = row['Distance']
                features['DistanceNorm'] = distance / 800.0  # Normalize to ~0-1
                
                # Box draw
                features['BoxDraw'] = row['Box'] / 8.0
                
                # Days since last race (approximate via history)
                if len(greyhound_hist) > 0:
                    most_recent = greyhound_hist.iloc[0]['MeetingDate']
                    days_since = pd.Timestamp(row['MeetingDate']).day - pd.Timestamp(most_recent).day
                    features['DaysSinceLastRace'] = max(1, min(days_since, 100)) / 100.0  # Cap at 100
                else:
                    features['DaysSinceLastRace'] = 0.5
            
            features['EntryID'] = row['EntryID']
            features['IsWinner'] = 1 if row['Position'] == '1' or row['Position'] == 1 else 0
            features['StartingPrice'] = row['StartingPrice']
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"Extracted {len(features_df)} feature vectors")
        
        return features_df
    
    def train_and_test(self, features_df, model_name):
        """Train model and test on $1.50-$2.00 range"""
        
        # Define features based on what's available
        base_features = [
            'BoxWinRate', 'AvgPositionLast3', 'WinRateLast3',
            'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'
        ]
        
        new_features = ['WeightClass', 'DistanceNorm', 'BoxDraw', 'DaysSinceLastRace']
        
        feature_cols = base_features + [f for f in new_features if f in features_df.columns]
        
        X = features_df[feature_cols]
        y = features_df['IsWinner']
        
        # Remove rows with NaN
        valid_idx = X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"  Training samples: {len(X)}, Features: {len(feature_cols)}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_loss = log_loss(y_test, model.predict_proba(X_test)[:, 1])
        
        # Feature importance
        importance_dict = dict(zip(feature_cols, model.feature_importances_))
        
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy: {test_acc:.1%}")
        print(f"  Test Log Loss: {test_loss:.4f}")
        
        return {
            'model': model,
            'feature_cols': feature_cols,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'importance': importance_dict
        }
    
    def backtest_on_1_5_to_2(self, model_info):
        """Backtest model on $1.50-$2.00 range"""
        
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        # Load test data for 2025
        query = """
        SELECT ge.EntryID, ge.StartingPrice, ge.Position, g.GreyhoundID,
               rm.MeetingDate, t.TrackName, r.Distance, ge.Box, ge.Weight
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= '2025-01-01' AND rm.MeetingDate <= '2025-11-30'
          AND ge.StartingPrice > 1.5 AND ge.StartingPrice <= 2.0
          AND ge.Position IS NOT NULL
        """
        
        test_races = pd.read_sql_query(query, self.conn)
        print(f"  Testing on {len(test_races)} races in $1.50-$2.00 range...")
        
        # Get predictions
        predictions = []
        for idx, row in test_races.iterrows():
            if idx % 50000 == 0:
                print(f"    {idx}/{len(test_races)}")
                sys.stdout.flush()
            
            # Get historical features
            hist_query = """
            SELECT ge.Position, ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths,
                   t.TrackName, r.Distance, ge.Weight
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE ge.GreyhoundID = ? AND rm.MeetingDate < ?
              AND ge.Position IS NOT NULL
            ORDER BY rm.MeetingDate DESC LIMIT 5
            """
            
            cursor = self.conn.cursor()
            cursor.execute(hist_query, (row['GreyhoundID'], row['MeetingDate']))
            hist = cursor.fetchall()
            
            if len(hist) < 5:
                continue
            
            # Build features (simplified for speed)
            features_dict = {}
            features_dict['BoxWinRate'] = 0.125
            
            last_3_pos = [int(h[0]) for h in hist[:3]]
            features_dict['AvgPositionLast3'] = sum(last_3_pos) / len(last_3_pos)
            features_dict['WinRateLast3'] = sum(1 for p in last_3_pos if p == 1) / len(last_3_pos)
            
            for i, h in enumerate(hist, 1):
                g_ot, m_ot, track = h[1], h[2], h[3]
                track_weight = self.get_track_tier_weight(track)
                recency_weight = [2.0, 1.5, 1.0, 1.0, 1.0][i-1]
                features_dict[f'GM_OT_ADJ_{i}'] = (float(g_ot or 0) + float(m_ot or 0)) * track_weight * recency_weight
            
            # New features if in model
            if 'WeightClass' in feature_cols:
                w = row['Weight']
                features_dict['WeightClass'] = 0 if w < 30 else (1 if w <= 33 else 2)
            if 'DistanceNorm' in feature_cols:
                features_dict['DistanceNorm'] = row['Distance'] / 800.0
            if 'BoxDraw' in feature_cols:
                features_dict['BoxDraw'] = row['Box'] / 8.0
            if 'DaysSinceLastRace' in feature_cols:
                features_dict['DaysSinceLastRace'] = 0.5  # Default
            
            # Get prediction
            X_pred = pd.DataFrame([features_dict])[feature_cols]
            pred = model.predict_proba(X_pred)[0][1]
            
            is_winner = 1 if str(row['Position']) == '1' else 0
            predictions.append({
                'Pred': pred,
                'Odds': row['StartingPrice'],
                'IsWinner': is_winner
            })
        
        pred_df = pd.DataFrame(predictions)
        
        # Analyze at 80% threshold
        high_conf = pred_df[pred_df['Pred'] >= 0.8]
        
        if len(high_conf) == 0:
            return None
        
        wins = high_conf['IsWinner'].sum()
        strike = wins / len(high_conf) * 100
        
        returned = (high_conf['IsWinner'] * high_conf['Odds']).sum() + (1 - high_conf['IsWinner']).sum()
        staked = len(high_conf)
        profit = returned - staked
        roi = (profit / staked * 100) if staked > 0 else 0
        
        return {
            'bets': len(high_conf),
            'wins': wins,
            'strike': strike,
            'roi': roi,
            'profit': profit
        }


# Run comprehensive tests
print("\n" + "="*100)
print("COMPREHENSIVE MODEL TESTING - DETAILED IMPACT ANALYSIS")
print("="*100)

tester = ModelVariantTester()

results_summary = []

# Test different recency weights
for config_name, weights in tester.recency_configs.items():
    print(f"\n{'='*100}")
    print(f"CONFIG: {config_name.upper()} - Recency Weights: {weights}")
    print(f"{'='*100}")
    
    features = tester.extract_training_data(weights, include_new_features=False)
    model_info = tester.train_and_test(features, f"recency_{config_name}")
    backtest = tester.backtest_on_1_5_to_2(model_info)
    
    if backtest:
        print(f"\n  BACKTEST RESULTS ($1.50-$2.00):")
        print(f"    Bets: {backtest['bets']}")
        print(f"    Strike Rate: {backtest['strike']:.1f}%")
        print(f"    ROI: {backtest['roi']:.2f}%")
        results_summary.append({
            'Model': f"Recency: {config_name}",
            'Bets': backtest['bets'],
            'Strike': f"{backtest['strike']:.1f}%",
            'ROI': f"{backtest['roi']:.2f}%"
        })

# Test with new features
print(f"\n{'='*100}")
print(f"CONFIG: WITH NEW FEATURES (Weight, Distance, Box, DaysSince) + Baseline Recency")
print(f"{'='*100}")

features = tester.extract_training_data([2.0, 1.5, 1.0, 1.0, 1.0], include_new_features=True)
model_info = tester.train_and_test(features, "with_new_features")
backtest = tester.backtest_on_1_5_to_2(model_info)

if backtest:
    print(f"\n  BACKTEST RESULTS ($1.50-$2.00):")
    print(f"    Bets: {backtest['bets']}")
    print(f"    Strike Rate: {backtest['strike']:.1f}%")
    print(f"    ROI: {backtest['roi']:.2f}%")
    results_summary.append({
        'Model': 'With New Features',
        'Bets': backtest['bets'],
        'Strike': f"{backtest['strike']:.1f}%",
        'ROI': f"{backtest['roi']:.2f}%"
    })

# Summary table
print(f"\n\n{'='*100}")
print("RESULTS SUMMARY")
print(f"{'='*100}\n")

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

print(f"\n{'='*100}")
print("BASELINE: Recency [2.0, 1.5, 1.0, 1.0, 1.0] = +1.21% ROI (284 bets, 63.7% strike)")
print(f"{'='*100}")

tester.conn.close()
