"""
HIGH EDGE DEEP DIVE
Investigating the Edge >= 20% signal that showed +6.0% ROI at 6.6 bets/day

The model doesn't need to be calibrated - it just needs to identify dogs where market is wrong
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Track tiers
METRO_TRACKS = ['Wentworth Park', 'The Meadows', 'Sandown Park', 'Albion Park', 'Cannington', 
                'Meadows (MEP)', 'Sandown (SAP)']
PROVINCIAL_TRACKS = ['Richmond', 'Bulli', 'Dapto', 'Goulburn', 'Maitland', 'Gosford', 
                     'Geelong', 'Warragul', 'Ballarat', 'Warrnambool', 'Sale', 
                     'Ipswich', 'Bundaberg', 'Capalaba', 'Mandurah', 'Murray Bridge', 'Angle Park',
                     'Shepparton', 'Bendigo', 'Horsham', 'Cranbourne']

def get_track_tier(track):
    if track in METRO_TRACKS:
        return 'Metro'
    elif track in PROVINCIAL_TRACKS:
        return 'Provincial'
    return 'Country'

def get_track_tier_num(track):
    if track in METRO_TRACKS:
        return 3
    elif track in PROVINCIAL_TRACKS:
        return 2
    return 1

log("="*80)
log("HIGH EDGE VALUE BETTING - Deep Analysis")
log("="*80)

# ============================================================================
# Load historical data for features
# ============================================================================
log("\nLoading data...")

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
hist_df['PositionNum'] = pd.to_numeric(hist_df['Position'], errors='coerce')

log(f"Loaded {len(hist_df):,} historical entries")

# ============================================================================
# Feature calculator
# ============================================================================
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
    
    career = prior.groupby('GreyhoundID').agg({
        'Winner': 'mean',
        'RaceNum': 'max'
    })
    career.columns = ['CareerWinRate', 'CareerStarts']
    career = career.reset_index()
    
    features = features.merge(career, on='GreyhoundID', how='left')
    features = features[features['Last5_Count'] >= 3]
    
    return features

# ============================================================================
# Load full 2025 data and split into train/test periods
# ============================================================================
log("\nLoading 2025 race data...")

race_query = """
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
  AND ge.StartingPrice > 0
  AND rm.MeetingDate >= '2025-01-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

all_2025 = pd.read_sql_query(race_query, conn)
all_2025['StartingPrice'] = pd.to_numeric(all_2025['StartingPrice'], errors='coerce')
all_2025['Winner'] = (all_2025['Position'] == '1').astype(int)
all_2025['MeetingDate'] = pd.to_datetime(all_2025['MeetingDate'])
all_2025['TrackTierNum'] = all_2025['TrackName'].apply(get_track_tier_num)
all_2025['TrackTier'] = all_2025['TrackName'].apply(get_track_tier)
all_2025['ImpliedProb'] = 1.0 / all_2025['StartingPrice']

log(f"Loaded {len(all_2025):,} entries for 2025")

# ============================================================================
# Walk-forward validation: Train on first 6 months, test on each subsequent month
# ============================================================================
log("\n" + "="*80)
log("WALK-FORWARD VALIDATION")
log("="*80)

model_features = [
    'Box', 'Distance', 'TrackTierNum',
    'Last5_G_OT', 'Last5_G_Split', 'Last5_GM_OT',
    'Last5_WinRate', 'Last5_Wins', 'Last5_AvgPos',
    'Last5_AvgTier', 'CareerWinRate', 'CareerStarts',
]

# Test on each month from July to November (5 months)
all_value_bets = []

for test_month in range(7, 12):  # July to November
    train_end = f'2025-{test_month:02d}-01'
    test_start = train_end
    test_end = f'2025-{test_month+1:02d}-01' if test_month < 12 else '2025-12-01'
    
    log(f"\nTraining on Jan-{test_month-1} 2025, testing on month {test_month}...")
    
    # Training data
    train_df = all_2025[all_2025['MeetingDate'] < train_end].copy()
    test_df = all_2025[(all_2025['MeetingDate'] >= test_start) & 
                       (all_2025['MeetingDate'] < test_end)].copy()
    
    if len(train_df) < 1000 or len(test_df) < 100:
        continue
    
    # Add features by week
    train_df['Week'] = train_df['MeetingDate'].dt.to_period('W').dt.start_time
    train_weeks = sorted(train_df['Week'].unique())
    
    train_results = []
    for week_start in train_weeks:
        week_end = week_start + timedelta(days=7)
        week_races = train_df[(train_df['MeetingDate'] >= week_start) & 
                              (train_df['MeetingDate'] < week_end)].copy()
        if len(week_races) == 0:
            continue
        
        dog_ids = week_races['GreyhoundID'].unique()
        dog_features = get_dog_features(hist_df, dog_ids, week_start, n_races=5)
        
        if len(dog_features) > 0:
            week_races = week_races.merge(dog_features, on='GreyhoundID', how='inner')
            train_results.append(week_races)
    
    train_with_features = pd.concat(train_results, ignore_index=True)
    
    # Add features to test data
    test_df['Week'] = test_df['MeetingDate'].dt.to_period('W').dt.start_time
    test_weeks = sorted(test_df['Week'].unique())
    
    test_results = []
    for week_start in test_weeks:
        week_end = week_start + timedelta(days=7)
        week_races = test_df[(test_df['MeetingDate'] >= week_start) & 
                             (test_df['MeetingDate'] < week_end)].copy()
        if len(week_races) == 0:
            continue
        
        dog_ids = week_races['GreyhoundID'].unique()
        dog_features = get_dog_features(hist_df, dog_ids, week_start, n_races=5)
        
        if len(dog_features) > 0:
            week_races = week_races.merge(dog_features, on='GreyhoundID', how='inner')
            test_results.append(week_races)
    
    test_with_features = pd.concat(test_results, ignore_index=True)
    
    # Train model
    train_clean = train_with_features.dropna(subset=model_features)
    test_clean = test_with_features.dropna(subset=model_features).copy()
    
    if len(train_clean) < 1000 or len(test_clean) < 100:
        continue
    
    X_train = train_clean[model_features]
    y_train = train_clean['Winner']
    X_test = test_clean[model_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    model.fit(X_train_scaled, y_train)
    
    test_clean['ModelProb'] = model.predict_proba(X_test_scaled)[:, 1]
    test_clean['Edge'] = test_clean['ModelProb'] - test_clean['ImpliedProb']
    test_clean['TestMonth'] = test_month
    
    all_value_bets.append(test_clean)
    log(f"  Training: {len(train_clean):,} | Test: {len(test_clean):,}")

# Combine all test months
combined = pd.concat(all_value_bets, ignore_index=True)
log(f"\nTotal test entries: {len(combined):,}")

# ============================================================================
# Analyze Edge >= 20% in detail
# ============================================================================
log("\n" + "="*80)
log("EDGE >= 20% DETAILED ANALYSIS")
log("="*80)

# Overall results at different edge thresholds
log("\nEdge threshold performance (walk-forward):")
log("-" * 90)

for edge in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    value_bets = combined[combined['Edge'] >= edge]
    if len(value_bets) >= 20:
        wins = value_bets['Winner'].sum()
        strike = wins / len(value_bets) * 100
        profit = (value_bets['Winner'] * value_bets['StartingPrice']).sum() - len(value_bets)
        roi = profit / len(value_bets) * 100
        avg_odds = value_bets['StartingPrice'].mean()
        days = (value_bets['MeetingDate'].max() - value_bets['MeetingDate'].min()).days + 1
        daily = len(value_bets) / days
        
        log(f"Edge >= {edge:.0%}: {len(value_bets):4d} ({daily:5.1f}/day) | Strike {strike:5.1f}% | ${avg_odds:5.2f} | ROI {roi:+6.1f}%")

# ============================================================================
# Edge >= 20% by month (stability check)
# ============================================================================
log("\n" + "="*80)
log("MONTHLY STABILITY CHECK (Edge >= 20%)")
log("="*80)

value_20 = combined[combined['Edge'] >= 0.20].copy()

for month in sorted(value_20['TestMonth'].unique()):
    month_data = value_20[value_20['TestMonth'] == month]
    wins = month_data['Winner'].sum()
    bets = len(month_data)
    strike = wins / bets * 100 if bets > 0 else 0
    profit = (month_data['Winner'] * month_data['StartingPrice']).sum() - bets
    roi = profit / bets * 100 if bets > 0 else 0
    avg_odds = month_data['StartingPrice'].mean()
    month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]
    
    log(f"{month_name} 2025: {bets:4d} bets | {wins:3d} wins | Strike {strike:5.1f}% | ${avg_odds:5.2f} | ROI {roi:+6.1f}%")

# ============================================================================
# Edge >= 20% by track tier
# ============================================================================
log("\n" + "="*80)
log("BY TRACK TIER (Edge >= 20%)")
log("="*80)

for tier in ['Metro', 'Provincial', 'Country']:
    tier_data = value_20[value_20['TrackTier'] == tier]
    if len(tier_data) >= 10:
        wins = tier_data['Winner'].sum()
        bets = len(tier_data)
        strike = wins / bets * 100
        profit = (tier_data['Winner'] * tier_data['StartingPrice']).sum() - bets
        roi = profit / bets * 100
        avg_odds = tier_data['StartingPrice'].mean()
        days = (tier_data['MeetingDate'].max() - tier_data['MeetingDate'].min()).days + 1
        daily = bets / days
        
        log(f"{tier:12s}: {bets:4d} ({daily:.1f}/day) | {wins:3d} wins | Strike {strike:5.1f}% | ${avg_odds:5.2f} | ROI {roi:+6.1f}%")

# ============================================================================
# Edge >= 20% by price range
# ============================================================================
log("\n" + "="*80)
log("BY PRICE RANGE (Edge >= 20%)")
log("="*80)

for low, high in [(5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]:
    price_data = value_20[(value_20['StartingPrice'] >= low) & (value_20['StartingPrice'] < high)]
    if len(price_data) >= 10:
        wins = price_data['Winner'].sum()
        bets = len(price_data)
        strike = wins / bets * 100
        profit = (price_data['Winner'] * price_data['StartingPrice']).sum() - bets
        roi = profit / bets * 100
        avg_odds = price_data['StartingPrice'].mean()
        
        log(f"${low:2d}-${high:3d}: {bets:4d} bets | {wins:3d} wins | Strike {strike:5.1f}% | ${avg_odds:5.2f} | ROI {roi:+6.1f}%")

# ============================================================================
# Find optimal Edge + Price combination
# ============================================================================
log("\n" + "="*80)
log("OPTIMAL EDGE + PRICE COMBINATIONS")
log("="*80)

results = []
for edge in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    for low, high in [(5, 15), (10, 20), (15, 30), (20, 40), (10, 30), (15, 50)]:
        filtered = combined[(combined['Edge'] >= edge) & 
                           (combined['StartingPrice'] >= low) & 
                           (combined['StartingPrice'] < high)]
        
        if len(filtered) >= 50:
            wins = filtered['Winner'].sum()
            bets = len(filtered)
            strike = wins / bets * 100
            profit = (filtered['Winner'] * filtered['StartingPrice']).sum() - bets
            roi = profit / bets * 100
            avg_odds = filtered['StartingPrice'].mean()
            days = (filtered['MeetingDate'].max() - filtered['MeetingDate'].min()).days + 1
            daily = bets / days
            
            results.append({
                'Edge': edge,
                'PriceRange': f'${low}-${high}',
                'Bets': bets,
                'Daily': daily,
                'Wins': wins,
                'Strike': strike,
                'AvgOdds': avg_odds,
                'Profit': profit,
                'ROI': roi
            })

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    results_df = results_df.sort_values('ROI', ascending=False)
    
    log("\nTop configurations by ROI (min 50 bets):")
    log("-" * 90)
    for _, row in results_df.head(15).iterrows():
        log(f"Edge>={row['Edge']:.0%} | {row['PriceRange']:8s} | {row['Bets']:4d} ({row['Daily']:.1f}/d) | "
            f"Strike {row['Strike']:.1f}% | ${row['AvgOdds']:.2f} | ROI {row['ROI']:+.1f}%")
    
    # Configurations in target range (3-10 bets/day)
    log("\nConfigurations with 3-10 bets/day:")
    log("-" * 90)
    target = results_df[(results_df['Daily'] >= 3) & (results_df['Daily'] <= 10)]
    target = target.sort_values('ROI', ascending=False)
    for _, row in target.head(10).iterrows():
        log(f"Edge>={row['Edge']:.0%} | {row['PriceRange']:8s} | {row['Bets']:4d} ({row['Daily']:.1f}/d) | "
            f"Strike {row['Strike']:.1f}% | ${row['AvgOdds']:.2f} | ROI {row['ROI']:+.1f}%")

log("\n" + "="*80)
log("ANALYSIS COMPLETE")
log("="*80)
