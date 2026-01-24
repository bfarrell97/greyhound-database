"""
MODEL V2: Build predictive model WITHOUT using market odds
Goal: Find genuine predictive signal independent of market

The problem with V1: The model was 65% reliant on ImpliedProb (odds),
essentially just copying the market. We need to find edge OVER the market.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("MODEL V2: Finding edge WITHOUT market odds")
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
# Load data with pre-calculated features
# ============================================================================
progress("\nLoading race data and calculating features...")

# This query calculates features directly in SQL for speed
query = """
WITH dog_history AS (
    SELECT 
        ge.GreyhoundID,
        ge.RaceID,
        rm.MeetingDate,
        ge.FinishTimeBenchmarkLengths as G_OT,
        ge.SplitBenchmarkLengths as G_Split,
        rm.MeetingAvgBenchmarkLengths as M_OT,
        CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END as Winner,
        CASE WHEN ge.Position IN ('1', '2', '3') THEN 1 ELSE 0 END as Placed,
        ge.Position,
        t.TrackName,
        ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as CareerRaceNum
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND t.TrackName NOT LIKE '%NZ%'
      AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
)
SELECT 
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.StartingPrice,
    ge.Position,
    ge.FinishTimeBenchmarkLengths as G_OT,
    ge.SplitBenchmarkLengths as G_Split,
    r.Distance,
    r.RaceNumber,
    rm.MeetingDate,
    rm.MeetingAvgBenchmarkLengths as M_OT,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2025-01-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

df = pd.read_sql_query(query, conn)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
df['Winner'] = (df['Position'] == '1').astype(int)
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['TrackTierNum'] = df['TrackName'].apply(get_track_tier_num)
df['ImpliedProb'] = 1.0 / df['StartingPrice']

progress(f"Loaded {len(df):,} entries from 2025")

# ============================================================================
# Calculate historical features (proper walk-forward)
# ============================================================================
progress("\nCalculating historical features for each greyhound...")

# Get all historical data for feature calculation
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
hist_df['Placed'] = hist_df['Position'].isin(['1', '2', '3']).astype(int)
hist_df['GM_OT'] = hist_df['G_OT'] - hist_df['M_OT']
hist_df['GM_Split'] = hist_df['G_Split'] - hist_df['M_Split']
hist_df['TrackTierNum'] = hist_df['TrackName'].apply(get_track_tier_num)

try:
    hist_df['PositionNum'] = pd.to_numeric(hist_df['Position'], errors='coerce')
except:
    hist_df['PositionNum'] = np.nan

progress(f"Loaded {len(hist_df):,} historical entries")

# Process week by week
df['Week'] = df['MeetingDate'].dt.to_period('W').dt.start_time
weeks = sorted(df['Week'].unique())
progress(f"Processing {len(weeks)} weeks...")

def get_dog_features(hist_df, dog_ids, cutoff_date, n_races=5):
    """Calculate features for dogs using data before cutoff_date"""
    
    # Filter to before cutoff
    prior = hist_df[(hist_df['MeetingDate'] < cutoff_date) & 
                    (hist_df['GreyhoundID'].isin(dog_ids))].copy()
    
    if len(prior) == 0:
        return pd.DataFrame()
    
    # Rank races within each dog
    prior['RaceNum'] = prior.groupby('GreyhoundID').cumcount(ascending=False) + 1
    
    # Last N races
    last_n = prior[prior['RaceNum'] <= n_races]
    
    # Calculate features
    features = last_n.groupby('GreyhoundID').agg({
        'G_OT': 'mean',
        'G_Split': 'mean',
        'GM_OT': 'mean',
        'GM_Split': 'mean',
        'Winner': ['mean', 'sum'],
        'Placed': 'mean',
        'PositionNum': 'mean',
        'RaceNum': 'count',
        'TrackTierNum': 'mean',  # Avg tier of recent races
    })
    
    # Flatten column names
    features.columns = ['Last5_G_OT', 'Last5_G_Split', 'Last5_GM_OT', 'Last5_GM_Split',
                        'Last5_WinRate', 'Last5_Wins', 'Last5_PlaceRate', 'Last5_AvgPos',
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
    
    # Filter to dogs with enough data
    features = features[features['Last5_Count'] >= 3]
    
    return features

# Process each week
all_results = []
for i, week_start in enumerate(weeks):
    week_end = week_start + timedelta(days=7)
    week_races = df[(df['MeetingDate'] >= week_start) & (df['MeetingDate'] < week_end)].copy()
    
    if len(week_races) == 0:
        continue
    
    if i % 4 == 0:
        progress(f"Processing week {i+1}/{len(weeks)}...", indent=1)
    
    # Get dog features using data BEFORE this week
    dog_ids = week_races['GreyhoundID'].unique()
    dog_features = get_dog_features(hist_df, dog_ids, week_start, n_races=5)
    
    if len(dog_features) == 0:
        continue
    
    # Merge
    week_races = week_races.merge(dog_features, on='GreyhoundID', how='inner')
    
    if len(week_races) > 0:
        all_results.append(week_races)

df = pd.concat(all_results, ignore_index=True)
progress(f"Final dataset: {len(df):,} entries with features")

# ============================================================================
# Build model WITHOUT market odds
# ============================================================================
progress("\n" + "=" * 100)
progress("Building model WITHOUT market odds...")

# Features that don't use market information
model_features = [
    'Box', 'Distance', 'TrackTierNum',
    'Last5_G_OT', 'Last5_G_Split', 'Last5_GM_OT', 'Last5_GM_Split',
    'Last5_WinRate', 'Last5_Wins', 'Last5_PlaceRate', 'Last5_AvgPos',
    'Last5_AvgTier', 'CareerWinRate', 'CareerStarts',
]

df_model = df.dropna(subset=model_features)
progress(f"Entries with all features: {len(df_model):,}")

# Time split
split_date = '2025-09-01'
train_df = df_model[df_model['MeetingDate'] < split_date]
test_df = df_model[df_model['MeetingDate'] >= split_date].copy()

progress(f"Training: {len(train_df):,} | Test: {len(test_df):,}")

X_train = train_df[model_features]
y_train = train_df['Winner']
X_test = test_df[model_features]
y_test = test_df['Winner']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
progress("Training Random Forest (no odds)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50, 
                            random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': model_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

progress("\nFeature Importance (No Odds Model):")
progress("-" * 60)
for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    progress(f"{row['Feature']:20s} | {row['Importance']:.3f} | {bar}")

# ============================================================================
# Evaluate
# ============================================================================
progress("\n" + "=" * 100)
progress("Evaluating model predictions...")

test_df['ModelProb'] = rf.predict_proba(X_test_scaled)[:, 1]

# Rank within each race
test_df['ModelRank'] = test_df.groupby('RaceID')['ModelProb'].rank(ascending=False, method='first')
test_df['MarketRank'] = test_df.groupby('RaceID')['StartingPrice'].rank(ascending=True, method='first')

def analyze_picks(df, rank_col, name, top_n=1):
    picks = df[df[rank_col] <= top_n]
    wins = picks['Winner'].sum()
    total = len(picks)
    strike = wins / total * 100 if total > 0 else 0
    profit = (picks['Winner'] * picks['StartingPrice']).sum() - total
    roi = profit / total * 100 if total > 0 else 0
    avg_odds = picks['StartingPrice'].mean()
    return total, wins, strike, roi, avg_odds

progress("\nTop-1 Picks Comparison:")
progress("-" * 70)

model_stats = analyze_picks(test_df, 'ModelRank', 'Model', 1)
market_stats = analyze_picks(test_df, 'MarketRank', 'Market', 1)

progress(f"Model (no odds): {model_stats[0]:,} bets | {model_stats[1]} wins | {model_stats[2]:.1f}% | ${model_stats[4]:.2f} | ROI {model_stats[3]:+.1f}%")
progress(f"Market favourite: {market_stats[0]:,} bets | {market_stats[1]} wins | {market_stats[2]:.1f}% | ${market_stats[4]:.2f} | ROI {market_stats[3]:+.1f}%")

# ============================================================================
# Find where model disagrees with market
# ============================================================================
progress("\n" + "=" * 100)
progress("Where does model disagree with market?")

model_top1 = test_df[test_df['ModelRank'] == 1].copy()
model_top1['IsFavourite'] = model_top1['MarketRank'] == 1

# Agree vs disagree
agree = model_top1[model_top1['IsFavourite']]
disagree = model_top1[~model_top1['IsFavourite']]

progress(f"\nModel agrees with market (picks favourite): {len(agree):,} races")
if len(agree) > 0:
    wins = agree['Winner'].sum()
    strike = wins / len(agree) * 100
    profit = (agree['Winner'] * agree['StartingPrice']).sum() - len(agree)
    roi = profit / len(agree) * 100
    avg_odds = agree['StartingPrice'].mean()
    progress(f"  -> {wins} wins | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

progress(f"\nModel disagrees (picks non-favourite): {len(disagree):,} races")
if len(disagree) > 0:
    wins = disagree['Winner'].sum()
    strike = wins / len(disagree) * 100
    profit = (disagree['Winner'] * disagree['StartingPrice']).sum() - len(disagree)
    roi = profit / len(disagree) * 100
    avg_odds = disagree['StartingPrice'].mean()
    progress(f"  -> {wins} wins | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")
    
    # Price breakdown of disagreements
    progress("\n  Disagreements by price range:")
    for low, high in [(2, 4), (3, 5), (4, 7), (5, 10), (7, 15)]:
        subset = disagree[(disagree['StartingPrice'] >= low) & (disagree['StartingPrice'] < high)]
        if len(subset) >= 20:
            wins = subset['Winner'].sum()
            strike = wins / len(subset) * 100
            profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
            roi = profit / len(subset) * 100
            daily = len(subset) / 91
            progress(f"    ${low}-${high}: {len(subset)} bets ({daily:.1f}/day) | {strike:.1f}% | ROI {roi:+.1f}%")

# ============================================================================
# Analyze by model confidence
# ============================================================================
progress("\n" + "=" * 100)
progress("Model confidence analysis...")

# High confidence = model gives much higher probability than average
model_top1['AvgProb'] = model_top1.groupby('RaceID')['ModelProb'].transform('mean')
model_top1['ProbAdvantage'] = model_top1['ModelProb'] - model_top1['AvgProb']

progress("\nBy probability advantage over field:")
for thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    subset = model_top1[model_top1['ProbAdvantage'] >= thresh]
    if len(subset) >= 30:
        wins = subset['Winner'].sum()
        strike = wins / len(subset) * 100
        profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
        roi = profit / len(subset) * 100
        avg_odds = subset['StartingPrice'].mean()
        daily = len(subset) / 91
        progress(f"  Prob advantage >= {thresh:.0%}: {len(subset)} ({daily:.1f}/day) | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

# ============================================================================
# Combine model + market
# ============================================================================
progress("\n" + "=" * 100)
progress("Combining model with market...")

# Model top pick that's ALSO favourite or 2nd favourite
progress("\nModel top pick by market position:")
for market_pos in [1, 2, 3]:
    subset = model_top1[model_top1['MarketRank'] == market_pos]
    if len(subset) >= 50:
        wins = subset['Winner'].sum()
        strike = wins / len(subset) * 100
        profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
        roi = profit / len(subset) * 100
        avg_odds = subset['StartingPrice'].mean()
        progress(f"  Model top + Market #{market_pos}: {len(subset)} | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

# Model top + top 2 market
subset = model_top1[model_top1['MarketRank'] <= 2]
if len(subset) >= 50:
    wins = subset['Winner'].sum()
    strike = wins / len(subset) * 100
    profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
    roi = profit / len(subset) * 100
    avg_odds = subset['StartingPrice'].mean()
    daily = len(subset) / 91
    progress(f"\n  Model top + Market Top-2: {len(subset)} ({daily:.1f}/day) | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

# ============================================================================
# Track tier analysis
# ============================================================================
progress("\n" + "=" * 100)
progress("By track tier (model top picks):")

for tier, tier_name in [(3, 'Metro'), (2, 'Provincial'), (1, 'Country')]:
    subset = model_top1[model_top1['TrackTierNum'] == tier]
    if len(subset) >= 50:
        wins = subset['Winner'].sum()
        strike = wins / len(subset) * 100
        profit = (subset['Winner'] * subset['StartingPrice']).sum() - len(subset)
        roi = profit / len(subset) * 100
        avg_odds = subset['StartingPrice'].mean()
        progress(f"  {tier_name:12s}: {len(subset)} bets | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("COMPLETE")
progress("=" * 100)
