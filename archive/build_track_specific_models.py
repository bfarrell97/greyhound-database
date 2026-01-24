"""
Build separate XGBoost models for Metro vs Country tracks
Theory: Different track types may have different predictive patterns
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle
import sys

class TrackSpecificModel:
    def __init__(self, track_tier):
        self.track_tier = track_tier
        self.model = None
        self.feature_columns = [
            'BoxWinRate', 'AvgPositionLast3', 'WinRateLast3',
            'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'
        ]
        
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

    def get_track_tier_weight(self, track_name):
        if track_name in self.METRO_TRACKS:
            return self.TRACK_WEIGHTS['metro']
        elif track_name in self.PROVINCIAL_TRACKS:
            return self.TRACK_WEIGHTS['provincial']
        else:
            return self.TRACK_WEIGHTS['country']

    def get_track_tier(self, track_name):
        if track_name in self.METRO_TRACKS:
            return 'metro'
        elif track_name in self.PROVINCIAL_TRACKS:
            return 'provincial'
        else:
            return 'country'

    def extract_training_data(self, start_date='2023-01-01', end_date='2025-05-31'):
        """Extract training data for specific track tier"""
        print(f"\nExtracting {self.track_tier.upper()} track training data...")
        
        conn = sqlite3.connect('greyhound_racing.db')
        
        # Get all races with complete data
        query = """
        SELECT DISTINCT
            ge.EntryID, g.GreyhoundName, g.GreyhoundID,
            rm.MeetingDate, t.TrackName, t.TrackID, r.RaceNumber,
            ge.Box, ge.Position, ge.StartingPrice,
            ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
            rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
            r.Distance, ge.Weight
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= ? AND rm.MeetingDate <= ?
          AND ge.Position IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
          AND t.TrackName NOT IN ('Addington', 'Cambridge')
        ORDER BY rm.MeetingDate DESC, r.RaceNumber DESC
        """
        
        races = pd.read_sql_query(query, conn, params=(start_date, end_date))
        print(f"Loaded {len(races)} entries in date range")
        
        # Filter by track tier
        races['TrackTier'] = races['TrackName'].apply(self.get_track_tier)
        races = races[races['TrackTier'] == self.track_tier]
        print(f"After filtering to {self.track_tier.upper()}: {len(races)} entries")
        
        if len(races) == 0:
            return None
        
        races['IsWinner'] = (races['Position'] == '1').astype(int)
        
        # Get historical races for each dog
        print("Fetching historical races...")
        hist_query = """
        SELECT ge.GreyhoundID, rm.MeetingDate, r.RaceNumber,
               ge.Position, t.TrackName,
               ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.Position IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
        ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
        """
        
        hist_data = pd.read_sql_query(hist_query, conn)
        hist_grouped = hist_data.groupby('GreyhoundID')
        
        # Extract features
        features_list = []
        for idx, row in races.iterrows():
            if idx % 50000 == 0:
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
            features['BoxWinRate'] = 0.125  # Simplified for now
            
            # Recent form
            last_3 = last_5.head(3)
            last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
            features['AvgPositionLast3'] = last_3_positions.mean()
            features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3) if len(last_3) > 0 else 0
            
            # GM_OT_ADJ with recency weighting
            recency_weights = [2.0, 1.5, 1.0, 1.0, 1.0]
            for i, (_, race) in enumerate(last_5.iterrows(), 1):
                g_ot_adj = float(race['FinishTimeBenchmarkLengths']) if pd.notna(race['FinishTimeBenchmarkLengths']) else 0.0
                m_ot_adj = float(race['MeetingAvgBenchmarkLengths']) if pd.notna(race['MeetingAvgBenchmarkLengths']) else 0.0
                
                track_weight = self.get_track_tier_weight(race['TrackName'])
                recency_weight = recency_weights[i - 1]
                features[f'GM_OT_ADJ_{i}'] = (g_ot_adj + m_ot_adj) * track_weight * recency_weight
            
            features['EntryID'] = row['EntryID']
            features['IsWinner'] = row['IsWinner']
            features['StartingPrice'] = row['StartingPrice']
            features_list.append(features)
        
        conn.close()
        
        features_df = pd.DataFrame(features_list)
        print(f"Extracted {len(features_df)} feature vectors")
        print(f"Winners: {features_df['IsWinner'].sum()}")
        
        return features_df

    def train(self, X, y):
        """Train XGBoost model"""
        print(f"\nTraining {self.track_tier.upper()} model...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Validation accuracy: {val_acc:.1%}")
        
        return self.model

    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved {self.track_tier} model to {filename}")


# Build models for each track tier
print("="*80)
print("BUILDING TRACK-SPECIFIC MODELS")
print("="*80)

for tier in ['metro', 'provincial', 'country']:
    model = TrackSpecificModel(tier)
    
    # Extract training data
    train_df = model.extract_training_data(start_date='2023-01-01', end_date='2025-05-31')
    
    if train_df is None or len(train_df) == 0:
        print(f"Skipping {tier} - insufficient data")
        continue
    
    # Prepare features
    X = train_df[model.feature_columns]
    y = train_df['IsWinner']
    
    # Train
    model.train(X, y)
    
    # Save
    model.save(f'greyhound_model_{tier}.pkl')

print("\n" + "="*80)
print("TRACK-SPECIFIC MODELS COMPLETE")
print("="*80)
print("\nNext step: Test these models on $1.50-$3.00 odds to compare performance")
