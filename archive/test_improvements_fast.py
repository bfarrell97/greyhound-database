"""
FAST VERSION: Test only the highest-impact scenarios
Focuses on aggressive recency + new features (skip moderate/balanced)
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import sys

class FastModelTester:
    """Rapid model testing focusing on high-impact changes"""
    
    def __init__(self):
        self.conn = sqlite3.connect('greyhound_racing.db')
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
    
    def get_track_weight(self, track_name):
        if track_name in self.METRO_TRACKS:
            return self.TRACK_WEIGHTS['metro']
        elif track_name in self.PROVINCIAL_TRACKS:
            return self.TRACK_WEIGHTS['provincial']
        else:
            return self.TRACK_WEIGHTS['country']
    
    def quick_extract(self, recency_weights, use_new_features):
        """Fast extraction for June-Nov 2025 test data (3 months instead of 2+ years)"""
        print(f"  Extracting features...")
        
        # Get races from test period
        query = """
        SELECT ge.EntryID, g.GreyhoundID, ge.Weight, ge.Box, ge.Position,
               rm.MeetingDate, t.TrackName, r.Distance
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate >= '2025-06-01' AND rm.MeetingDate <= '2025-11-30'
          AND ge.Position IS NOT NULL
        ORDER BY rm.MeetingDate
        LIMIT 50000
        """
        
        races = pd.read_sql_query(query, self.conn)
        print(f"    Loaded {len(races)} entries from test period")
        
        features_list = []
        skipped = 0
        
        for idx, row in races.iterrows():
            # Get historical races for this dog
            self.cursor = self.conn.cursor()
            self.cursor.execute("""
                SELECT ge.Position, ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths, t.TrackName
                FROM GreyhoundEntries ge
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                WHERE ge.GreyhoundID = ? AND rm.MeetingDate < ?
                  AND ge.Position IS NOT NULL
                ORDER BY rm.MeetingDate DESC LIMIT 5
            """, (row['GreyhoundID'], row['MeetingDate']))
            
            hist = self.cursor.fetchall()
            if len(hist) < 5:
                skipped += 1
                continue
            
            # Build features
            features = {}
            features['BoxWinRate'] = 0.125
            
            # Recent form
            last_3_pos = [int(h[0]) for h in hist[:3] if str(h[0]).isdigit()]
            if not last_3_pos:
                skipped += 1
                continue
            
            features['AvgPositionLast3'] = sum(last_3_pos) / len(last_3_pos)
            features['WinRateLast3'] = sum(1 for p in last_3_pos if p == 1) / len(last_3_pos)
            
            # GM_OT_ADJ with recency
            for i, h in enumerate(hist, 1):
                g_ot = float(h[1]) if h[1] else 0
                m_ot = float(h[2]) if h[2] else 0
                track_weight = self.get_track_weight(h[3])
                features[f'GM_OT_ADJ_{i}'] = (g_ot + m_ot) * track_weight * recency_weights[i-1]
            
            if use_new_features:
                w = row['Weight']
                features['WeightClass'] = 0 if w < 30 else (1 if w <= 33 else 2)
                features['BoxDraw'] = row['Box'] / 8.0
            
            features['IsWinner'] = 1 if str(row['Position']) == '1' else 0
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        print(f"    Extracted {len(features_df)} valid examples ({skipped} skipped)")
        
        return features_df
    
    def test_scenario(self, name, recency_weights, use_new_features):
        """Test a single scenario"""
        print(f"\n{'='*90}")
        print(f"TESTING: {name}")
        print(f"  Recency: {recency_weights}")
        print(f"  New Features: {use_new_features}")
        print(f"{'='*90}")
        
        # Extract
        features = self.quick_extract(recency_weights, use_new_features)
        
        # Define features
        base_features = ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3',
                         'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5']
        
        new_feats = ['WeightClass', 'BoxDraw']
        feature_cols = base_features + [f for f in new_feats if f in features.columns]
        
        # Clean data
        X = features[feature_cols].fillna(0)
        y = features['IsWinner']
        
        # Train
        print(f"  Training on {len(X)} examples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=5, learning_rate=0.15,
            subsample=0.8, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy: {test_acc:.1%}")
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top Features: {', '.join([f'{f[0]}:{f[1]:.1%}' for f in top_features])}")
        
        return {
            'name': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': model,
            'feature_cols': feature_cols,
            'importance': importance
        }


# Run tests
print("\n" + "="*90)
print("FAST MODEL IMPROVEMENT TESTING")
print("="*90)

tester = FastModelTester()

scenarios = [
    ("BASELINE", [2.0, 1.5, 1.0, 1.0, 1.0], False),
    ("AGGRESSIVE RECENCY", [3.0, 1.5, 0.8, 0.5, 0.3], False),
    ("BASELINE + NEW FEATURES", [2.0, 1.5, 1.0, 1.0, 1.0], True),
    ("AGGRESSIVE + NEW FEATURES", [3.0, 1.5, 0.8, 0.5, 0.3], True),
]

results = []
for name, weights, use_new in scenarios:
    result = tester.test_scenario(name, weights, use_new)
    results.append(result)

print(f"\n\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}\n")

summary_df = pd.DataFrame({
    'Scenario': [r['name'] for r in results],
    'Train Acc': [f"{r['train_acc']:.1%}" for r in results],
    'Test Acc': [f"{r['test_acc']:.1%}" for r in results]
})

print(summary_df.to_string(index=False))

print("\n" + "="*90)
print("NEXT STEP: Based on which performs best, retrain full model on 2023-2025 data")
print("="*90)

tester.conn.close()
