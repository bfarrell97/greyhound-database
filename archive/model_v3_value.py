"""
MODEL V3: Finding VALUE bets
Instead of trying to pick winners, find dogs that are undervalued by the market

Strategy: Compare model's probability estimate vs market implied probability
Bet when model thinks dog is significantly better than odds suggest
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("MODEL V3: Finding VALUE bets")
progress("=" * 100)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Track tiers
METRO_TRACKS = ['Wentworth Park', 'The Meadows', 'Sandown Park', 'Albion Park', 'Cannington', 
                'Meadows (MEP)', 'Sandown (SAP)']
PROVINCIAL_TRACKS = ['Richmond', 'Bulli', 'Dapto', 'Goulburn', 'Maitland', 'Gosford', 
                     'Geelong', 'Warragul', 'Ballarat', 'Warrnambool', 'Sale', 
                     'Ipswich', 'Bundaberg', 'Capalaba', 'Mandurah', 'Murray Bridge', 'Angle Park',
                     'Shepparton', 'Bendigo', 'Horsham', 'Cranbourne']

def get_track_tier_num(track_name):
    if track_name in METRO_TRACKS:
        return 3
    elif track_name in PROVINCIAL_TRACKS:
        return 2
    else:
        return 1

# ============================================================================
# Load data
# ============================================================================
progress("\nLoading data...")

hist_query = """
SELECT 
    ge.GreyhoundID,
    rm.MeetingDate,
    t.TrackName,
    ge.FinishTimeBenchmarkLengths as G_OT,
    ge.SplitBenchmarkLengths as G_Split,
    rm.MeetingAvgBenchmarkLengths as M_OT,
    rm.MeetingSplitAvgBenchmarkLengths as M_Split,
    ge.Position,
    r.Distance,
    ge.Box
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
ORDER BY ge.GreyhoundID, rm.MeetingDate
"""

hist_df = pd.read_sql_query(hist_query, conn)
hist_df['MeetingDate'] = pd.to_datetime(hist_df['MeetingDate'])
hist_df['Winner'] = (hist_df['Position'] == '1').astype(int)
hist_df['GM_OT'] = hist_df['G_OT'] - hist_df['M_OT']
hist_df['GM_Split'] = hist_df['G_Split'] - hist_df['M_Split']
hist_df['TrackTierNum'] = hist_df['TrackName'].apply(get_track_tier_num)
try:
    hist_df['PositionNum'] = pd.to_numeric(hist_df['Position'], errors='coerce')
except:
    pass

progress(f"Loaded {len(hist_df):,} historical entries")

# Test period data
test_query = """
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    r.Distance,
    rm.MeetingDate,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2025-09-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

test_df = pd.read_sql_query(test_query, conn)
test_df['StartingPrice'] = pd.to_numeric(test_df['StartingPrice'], errors='coerce')
test_df['Winner'] = (test_df['Position'] == '1').astype(int)
test_df['MeetingDate'] = pd.to_datetime(test_df['MeetingDate'])
test_df['TrackTierNum'] = test_df['TrackName'].apply(get_track_tier_num)
test_df['ImpliedProb'] = 1.0 / test_df['StartingPrice']

progress(f"Test period: {len(test_df):,} entries (Sept-Nov 2025)")

# ============================================================================
# Calculate features
# ============================================================================
progress("\nCalculating features...")

def get_dog_features(hist_df, dog_ids, cutoff_date, n_races=5):
    prior = hist_df[(hist_df['MeetingDate'] < cutoff_date) & 
                    (hist_df['GreyhoundID'].isin(dog_ids))].copy()
    
    if len(prior) == 0:
        return pd.DataFrame()
    
    prior['RaceNum'] = prior.groupby('GreyhoundID').cumcount(ascending=False) + 1
    last_n = prior[prior['RaceNum'] <= n_races]
    
    features = last_n.groupby('GreyhoundID').agg({
        'G_OT': 'mean',
        'G_Split': 'mean',
        'GM_OT': 'mean',
        'Winner': ['mean', 'sum'],
        'PositionNum': 'mean',
        'RaceNum': 'count',
        'TrackTierNum': 'mean',
    })
    
    features.columns = ['Last5_G_OT', 'Last5_G_Split', 'Last5_GM_OT',
                        'Last5_WinRate', 'Last5_Wins', 'Last5_AvgPos',
                        'Last5_Count', 'Last5_AvgTier']
    
    features = features.reset_index()
    
    # Career stats
    career = prior.groupby('GreyhoundID').agg({
        'Winner': 'mean',
        'RaceNum': 'max'
    })
    career.columns = ['CareerWinRate', 'CareerStarts']
    career = career.reset_index()
    
    features = features.merge(career, on='GreyhoundID', how='left')
    features = features[features['Last5_Count'] >= 3]
    
    return features

# Get training data (Jan-Aug 2025)
progress("Building training data (Jan-Aug 2025)...")

train_query = """
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    r.Distance,
    rm.MeetingDate,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2025-01-01'
  AND rm.MeetingDate < '2025-09-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

train_df = pd.read_sql_query(train_query, conn)
train_df['StartingPrice'] = pd.to_numeric(train_df['StartingPrice'], errors='coerce')
train_df['Winner'] = (train_df['Position'] == '1').astype(int)
train_df['MeetingDate'] = pd.to_datetime(train_df['MeetingDate'])
train_df['TrackTierNum'] = train_df['TrackName'].apply(get_track_tier_num)
train_df['ImpliedProb'] = 1.0 / train_df['StartingPrice']

progress(f"Train period: {len(train_df):,} entries")

# Add features to training data
train_df['Week'] = train_df['MeetingDate'].dt.to_period('W').dt.start_time
train_weeks = sorted(train_df['Week'].unique())

train_results = []
for week_start in train_weeks:
    week_end = week_start + timedelta(days=7)
    week_races = train_df[(train_df['MeetingDate'] >= week_start) & (train_df['MeetingDate'] < week_end)].copy()
    
    if len(week_races) == 0:
        continue
    
    dog_ids = week_races['GreyhoundID'].unique()
    dog_features = get_dog_features(hist_df, dog_ids, week_start, n_races=5)
    
    if len(dog_features) > 0:
        week_races = week_races.merge(dog_features, on='GreyhoundID', how='inner')
        train_results.append(week_races)

train_df = pd.concat(train_results, ignore_index=True)
progress(f"Training data with features: {len(train_df):,}")

# Add features to test data
test_df['Week'] = test_df['MeetingDate'].dt.to_period('W').dt.start_time
test_weeks = sorted(test_df['Week'].unique())

test_results = []
for week_start in test_weeks:
    week_end = week_start + timedelta(days=7)
    week_races = test_df[(test_df['MeetingDate'] >= week_start) & (test_df['MeetingDate'] < week_end)].copy()
    
    if len(week_races) == 0:
        continue
    
    dog_ids = week_races['GreyhoundID'].unique()
    dog_features = get_dog_features(hist_df, dog_ids, week_start, n_races=5)
    
    if len(dog_features) > 0:
        week_races = week_races.merge(dog_features, on='GreyhoundID', how='inner')
        test_results.append(week_races)

test_df = pd.concat(test_results, ignore_index=True)
progress(f"Test data with features: {len(test_df):,}")

# ============================================================================
# Train calibrated model
# ============================================================================
progress("\n" + "=" * 100)
progress("Training calibrated probability model...")

model_features = [
    'Box', 'Distance', 'TrackTierNum',
    'Last5_G_OT', 'Last5_G_Split', 'Last5_GM_OT',
    'Last5_WinRate', 'Last5_Wins', 'Last5_AvgPos',
    'Last5_AvgTier', 'CareerWinRate', 'CareerStarts',
]

train_clean = train_df.dropna(subset=model_features)
test_clean = test_df.dropna(subset=model_features).copy()

X_train = train_clean[model_features]
y_train = train_clean['Winner']
X_test = test_clean[model_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with calibration for better probability estimates
progress("Training Gradient Boosting with calibration...")
base_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                         learning_rate=0.1, random_state=42)
model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
model.fit(X_train_scaled, y_train)

# Get probabilities
test_clean['ModelProb'] = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================================
# Find VALUE: Model Prob > Implied Prob
# ============================================================================
progress("\n" + "=" * 100)
progress("Finding VALUE bets (Model Prob > Market Implied Prob)...")

test_clean['Edge'] = test_clean['ModelProb'] - test_clean['ImpliedProb']
test_clean['EdgePct'] = test_clean['Edge'] / test_clean['ImpliedProb'] * 100

# Value bets at different thresholds
progress("\nValue bets by edge threshold:")
progress("-" * 80)

for edge in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    value_bets = test_clean[test_clean['Edge'] >= edge]
    if len(value_bets) >= 20:
        wins = value_bets['Winner'].sum()
        strike = wins / len(value_bets) * 100
        profit = (value_bets['Winner'] * value_bets['StartingPrice']).sum() - len(value_bets)
        roi = profit / len(value_bets) * 100
        avg_odds = value_bets['StartingPrice'].mean()
        daily = len(value_bets) / 91
        
        # Expected ROI based on edge
        expected_strike = value_bets['ModelProb'].mean() * 100
        expected_return = (value_bets['ModelProb'] * value_bets['StartingPrice']).mean()
        
        progress(f"Edge >= {edge:.0%}: {len(value_bets):4d} ({daily:4.1f}/day) | Model expects {expected_strike:.1f}% | Actual {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

# ============================================================================
# Value bets by price range
# ============================================================================
progress("\n" + "=" * 100)
progress("Value bets (Edge >= 5%) by price range:")

for low, high in [(1.5, 2.5), (2.0, 3.5), (2.5, 4.0), (3.0, 5.0), (4.0, 7.0), (6.0, 12.0)]:
    value_bets = test_clean[(test_clean['Edge'] >= 0.05) & 
                            (test_clean['StartingPrice'] >= low) & 
                            (test_clean['StartingPrice'] < high)]
    if len(value_bets) >= 15:
        wins = value_bets['Winner'].sum()
        strike = wins / len(value_bets) * 100
        profit = (value_bets['Winner'] * value_bets['StartingPrice']).sum() - len(value_bets)
        roi = profit / len(value_bets) * 100
        daily = len(value_bets) / 91
        progress(f"  ${low}-${high}: {len(value_bets):4d} ({daily:.1f}/day) | {strike:.1f}% | ROI {roi:+.1f}%")

# ============================================================================
# Best value configurations
# ============================================================================
progress("\n" + "=" * 100)
progress("Testing all filter combinations...")

results = []

# Test combinations of edge threshold + price range + track tier
for edge_thresh in [0.03, 0.05, 0.07, 0.10]:
    for low, high in [(1.5, 3.0), (2.0, 4.0), (2.5, 5.0), (3.0, 6.0), (4.0, 8.0)]:
        for tier in [None, 3, 2, 1]:  # None = all tiers
            if tier is None:
                subset = test_clean[(test_clean['Edge'] >= edge_thresh) & 
                                   (test_clean['StartingPrice'] >= low) & 
                                   (test_clean['StartingPrice'] < high)]
                tier_name = "All"
            else:
                subset = test_clean[(test_clean['Edge'] >= edge_thresh) & 
                                   (test_clean['StartingPrice'] >= low) & 
                                   (test_clean['StartingPrice'] < high) &
                                   (test_clean['TrackTierNum'] == tier)]
                tier_name = {3: 'Metro', 2: 'Provincial', 1: 'Country'}[tier]
            
            if len(subset) >= 30:
                wins = subset['Winner'].sum()
                strike = wins / len(subset) * 100
                profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
                roi = profit / len(subset) * 100
                daily = len(subset) / 91
                
                if daily >= 0.5:  # At least 1 bet every 2 days
                    results.append({
                        'Edge': edge_thresh,
                        'Price': f'${low}-${high}',
                        'Tier': tier_name,
                        'Bets': len(subset),
                        'Daily': daily,
                        'Strike': strike,
                        'ROI': roi
                    })

# Sort by ROI
results_df = pd.DataFrame(results).sort_values('ROI', ascending=False)

progress("\nTop 20 configurations by ROI:")
progress("-" * 90)
for _, row in results_df.head(20).iterrows():
    progress(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tier']:10s} | {row['Bets']:4d} ({row['Daily']:.1f}/day) | {row['Strike']:.1f}% | ROI {row['ROI']:+.1f}%")

# ============================================================================
# Best strike rate configurations
# ============================================================================
progress("\n" + "=" * 100)
progress("Configurations with highest strike rate (for betting strategy):")

results_by_strike = results_df[results_df['Bets'] >= 50].sort_values('Strike', ascending=False)

progress("-" * 90)
for _, row in results_by_strike.head(15).iterrows():
    progress(f"Edge>={row['Edge']:.0%} | {row['Price']:10s} | {row['Tier']:10s} | {row['Bets']:4d} ({row['Daily']:.1f}/day) | Strike: {row['Strike']:.1f}% | ROI {row['ROI']:+.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("COMPLETE")
progress("=" * 100)
