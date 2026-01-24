"""
COMPREHENSIVE FEATURE ANALYSIS
Find what data actually predicts winners in greyhound racing

Step 1: Load all available data
Step 2: Calculate correlation with winning
Step 3: Build predictive model
Step 4: Evaluate out-of-sample performance
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("=" * 100)
progress("COMPREHENSIVE FEATURE ANALYSIS - What predicts winners?")
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

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'Metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'Provincial'
    else:
        return 'Country'

def get_track_tier_num(track_name):
    if track_name in METRO_TRACKS:
        return 3
    elif track_name in PROVINCIAL_TRACKS:
        return 2
    else:
        return 1

# ============================================================================
# STEP 1: Load comprehensive race data
# ============================================================================
progress("\nSTEP 1: Loading all race data...")

# Get races with all available fields
query = """
SELECT 
    ge.EntryID,
    ge.GreyhoundID,
    ge.RaceID,
    ge.Box,
    ge.Weight,
    ge.Position,
    ge.StartingPrice,
    ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
    ge.SplitBenchmarkLengths as G_Split_ADJ,
    ge.InRun,
    r.Distance,
    r.RaceNumber,
    r.Grade,
    rm.MeetingDate,
    rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
    rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
    t.TrackName
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.Position NOT IN ('DNF', 'SCR', '')
  AND ge.StartingPrice IS NOT NULL
  AND rm.MeetingDate >= '2024-01-01'
  AND rm.MeetingDate < '2025-12-01'
  AND t.TrackName NOT LIKE '%NZ%'
  AND t.TrackName NOT IN ('Hobart', 'Launceston', 'Devonport')
"""

df = pd.read_sql_query(query, conn)
progress(f"Loaded {len(df):,} race entries")

# Convert types
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
df['Winner'] = (df['Position'] == '1').astype(int)
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])

# Add track tier
df['TrackTier'] = df['TrackName'].apply(get_track_tier)
df['TrackTierNum'] = df['TrackName'].apply(get_track_tier_num)

# Calculate G/M adjustments
df['GM_OT_ADJ'] = df['G_OT_ADJ'] - df['M_OT_ADJ']
df['GM_Split_ADJ'] = df['G_Split_ADJ'] - df['M_Split_ADJ']

# Implied probability from odds
df['ImpliedProb'] = 1.0 / df['StartingPrice']

# InRun position at first split
df['InRunFirst'] = df['InRun'].astype(str).str[0]
df['InRunFirst'] = pd.to_numeric(df['InRunFirst'], errors='coerce')

progress(f"Date range: {df['MeetingDate'].min()} to {df['MeetingDate'].max()}")
progress(f"Winners: {df['Winner'].sum():,} ({df['Winner'].mean()*100:.1f}%)")

# ============================================================================
# STEP 2: Calculate historical features for each dog
# ============================================================================
progress("\nSTEP 2: Calculating historical features for each entry...")

# Sort by date
df = df.sort_values(['GreyhoundID', 'MeetingDate'])

# For each entry, calculate features from PRIOR races only
# We need to do this carefully to avoid look-ahead bias

def calculate_dog_features(group):
    """Calculate rolling features from prior races for each dog"""
    n = len(group)
    
    # Initialize feature columns
    features = {
        'Last5_G_OT_ADJ_Avg': [np.nan] * n,
        'Last5_G_Split_ADJ_Avg': [np.nan] * n,
        'Last5_GM_OT_ADJ_Avg': [np.nan] * n,
        'Last5_GM_Split_ADJ_Avg': [np.nan] * n,
        'Last5_WinRate': [np.nan] * n,
        'Last5_PlaceRate': [np.nan] * n,
        'Last5_AvgPosition': [np.nan] * n,
        'Last10_WinRate': [np.nan] * n,
        'Career_WinRate': [np.nan] * n,
        'Career_Starts': [0] * n,
        'Days_Since_Last_Race': [np.nan] * n,
        'Last5_AvgOdds': [np.nan] * n,
        'Last_Race_Win': [np.nan] * n,
        'Last_Race_Position': [np.nan] * n,
        # Track-specific
        'Last5_Metro_G_OT_ADJ': [np.nan] * n,
        'Last5_Provincial_G_OT_ADJ': [np.nan] * n,
        'Last5_Country_G_OT_ADJ': [np.nan] * n,
    }
    
    for i in range(n):
        # Get prior races (before current race)
        prior = group.iloc[:i]
        
        if len(prior) == 0:
            continue
        
        # Career stats
        features['Career_Starts'][i] = len(prior)
        features['Career_WinRate'][i] = prior['Winner'].mean() * 100
        
        # Days since last race
        if i > 0:
            features['Days_Since_Last_Race'][i] = (group.iloc[i]['MeetingDate'] - group.iloc[i-1]['MeetingDate']).days
            features['Last_Race_Win'][i] = prior.iloc[-1]['Winner']
            try:
                features['Last_Race_Position'][i] = int(prior.iloc[-1]['Position'])
            except:
                pass
        
        # Last 5 races
        last5 = prior.tail(5)
        if len(last5) >= 3:  # Need at least 3 races
            features['Last5_G_OT_ADJ_Avg'][i] = last5['G_OT_ADJ'].mean()
            features['Last5_G_Split_ADJ_Avg'][i] = last5['G_Split_ADJ'].mean()
            features['Last5_GM_OT_ADJ_Avg'][i] = last5['GM_OT_ADJ'].mean()
            features['Last5_GM_Split_ADJ_Avg'][i] = last5['GM_Split_ADJ'].mean()
            features['Last5_WinRate'][i] = last5['Winner'].mean() * 100
            features['Last5_PlaceRate'][i] = (last5['Position'].isin(['1', '2', '3'])).mean() * 100
            features['Last5_AvgOdds'][i] = last5['StartingPrice'].mean()
            
            try:
                positions = pd.to_numeric(last5['Position'], errors='coerce')
                features['Last5_AvgPosition'][i] = positions.mean()
            except:
                pass
        
        # Last 10 races
        last10 = prior.tail(10)
        if len(last10) >= 5:
            features['Last10_WinRate'][i] = last10['Winner'].mean() * 100
        
        # Track tier specific (last 5 at each tier)
        for tier in ['Metro', 'Provincial', 'Country']:
            tier_races = prior[prior['TrackTier'] == tier].tail(5)
            if len(tier_races) >= 2:
                col_name = f'Last5_{tier}_G_OT_ADJ'
                features[col_name][i] = tier_races['G_OT_ADJ'].mean()
    
    return pd.DataFrame(features, index=group.index)

progress("Calculating features (this may take a few minutes)...")

# Process in chunks by greyhound
feature_dfs = []
unique_dogs = df['GreyhoundID'].unique()
total_dogs = len(unique_dogs)

for idx, gid in enumerate(unique_dogs):
    if idx % 5000 == 0:
        progress(f"Processing dog {idx:,}/{total_dogs:,}...", indent=1)
    
    dog_data = df[df['GreyhoundID'] == gid].copy()
    dog_features = calculate_dog_features(dog_data)
    feature_dfs.append(dog_features)

features_df = pd.concat(feature_dfs)
df = df.join(features_df)

progress(f"Features calculated for {len(df):,} entries")

# ============================================================================
# STEP 3: Analyze feature correlations with winning
# ============================================================================
progress("\nSTEP 3: Analyzing feature correlations with winning...")

# Focus on entries with sufficient data
df_analysis = df.dropna(subset=['Last5_G_OT_ADJ_Avg', 'Last5_WinRate']).copy()
progress(f"Entries with features: {len(df_analysis):,}")

# List of features to analyze
feature_cols = [
    'Box', 'Weight', 'Distance', 'TrackTierNum', 'ImpliedProb',
    'G_OT_ADJ', 'G_Split_ADJ', 'GM_OT_ADJ', 'GM_Split_ADJ',
    'Last5_G_OT_ADJ_Avg', 'Last5_G_Split_ADJ_Avg', 
    'Last5_GM_OT_ADJ_Avg', 'Last5_GM_Split_ADJ_Avg',
    'Last5_WinRate', 'Last5_PlaceRate', 'Last5_AvgPosition',
    'Last10_WinRate', 'Career_WinRate', 'Career_Starts',
    'Days_Since_Last_Race', 'Last5_AvgOdds',
    'Last_Race_Win', 'Last_Race_Position',
]

# Calculate correlations
correlations = []
for col in feature_cols:
    if col in df_analysis.columns:
        valid = df_analysis[[col, 'Winner']].dropna()
        if len(valid) > 1000:
            corr = valid[col].corr(valid['Winner'])
            correlations.append((col, corr, len(valid)))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

progress("\nFeature correlations with winning (sorted by strength):")
progress("-" * 60)
for col, corr, count in correlations:
    direction = "+" if corr > 0 else ""
    progress(f"{col:30s} | r = {direction}{corr:.4f} | n = {count:,}")

# ============================================================================
# STEP 4: Build predictive model
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 4: Building predictive model...")

# Prepare data for modeling
model_features = [
    'Box', 'ImpliedProb', 'TrackTierNum', 'Distance',
    'Last5_G_OT_ADJ_Avg', 'Last5_G_Split_ADJ_Avg',
    'Last5_WinRate', 'Last5_PlaceRate', 'Last5_AvgPosition',
    'Career_WinRate', 'Career_Starts',
    'Days_Since_Last_Race', 'Last5_AvgOdds',
    'Last_Race_Win',
]

# Filter to recent data with all features
df_model = df_analysis[df_analysis['MeetingDate'] >= '2025-01-01'].copy()
df_model = df_model.dropna(subset=model_features)

progress(f"Modeling dataset: {len(df_model):,} entries from 2025")

# Time-based train/test split (train on earlier, test on later)
split_date = '2025-09-01'
train_df = df_model[df_model['MeetingDate'] < split_date]
test_df = df_model[df_model['MeetingDate'] >= split_date]

progress(f"Training set: {len(train_df):,} (before {split_date})")
progress(f"Test set: {len(test_df):,} (from {split_date})")

X_train = train_df[model_features]
y_train = train_df['Winner']
X_test = test_df[model_features]
y_test = test_df['Winner']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
progress("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': model_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

progress("\nFeature Importance (Random Forest):")
progress("-" * 50)
for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    progress(f"{row['Feature']:25s} | {row['Importance']:.3f} | {bar}")

# ============================================================================
# STEP 5: Evaluate model predictions
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 5: Evaluating model on test set...")

# Get predictions
test_df = test_df.copy()
test_df['Pred_Prob'] = rf.predict_proba(X_test_scaled)[:, 1]

# Rank predictions within each race
test_df['Pred_Rank'] = test_df.groupby('RaceID')['Pred_Prob'].rank(ascending=False, method='first')

# Calculate market rank
test_df['Market_Rank'] = test_df.groupby('RaceID')['StartingPrice'].rank(ascending=True, method='first')

# Accuracy at different thresholds
progress("\nModel vs Market - Picking Winners:")
progress("-" * 70)

def analyze_picks(df, rank_col, name):
    """Analyze performance of top picks"""
    results = []
    for top_n in [1, 2, 3]:
        picks = df[df[rank_col] <= top_n]
        wins = picks['Winner'].sum()
        total = len(picks)
        strike = wins / total * 100 if total > 0 else 0
        
        # ROI
        profit = (picks['Winner'] * picks['StartingPrice']).sum() - total
        roi = profit / total * 100 if total > 0 else 0
        
        avg_odds = picks['StartingPrice'].mean()
        results.append((top_n, total, wins, strike, roi, avg_odds))
    
    progress(f"\n{name}:")
    for top_n, total, wins, strike, roi, avg_odds in results:
        progress(f"  Top-{top_n}: {total:,} bets | {wins} wins | {strike:.1f}% strike | ${avg_odds:.2f} avg | ROI {roi:+.1f}%")

analyze_picks(test_df, 'Pred_Rank', 'Model Predictions')
analyze_picks(test_df, 'Market_Rank', 'Market (Favourites)')

# ============================================================================
# STEP 6: Find model edge over market
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 6: Finding where model beats market...")

# Cases where model top pick differs from market favourite
model_top1 = test_df[test_df['Pred_Rank'] == 1].copy()
model_top1['IsFavourite'] = model_top1['Market_Rank'] == 1

# Model's non-favourite picks
non_fav_picks = model_top1[~model_top1['IsFavourite']]
if len(non_fav_picks) > 0:
    wins = non_fav_picks['Winner'].sum()
    strike = wins / len(non_fav_picks) * 100
    profit = (non_fav_picks['Winner'] * non_fav_picks['StartingPrice']).sum() - len(non_fav_picks)
    roi = profit / len(non_fav_picks) * 100
    avg_odds = non_fav_picks['StartingPrice'].mean()
    progress(f"\nModel's Top Pick when NOT favourite:")
    progress(f"  {len(non_fav_picks)} bets | {wins} wins | {strike:.1f}% strike | ${avg_odds:.2f} avg | ROI {roi:+.1f}%")

# Model's favourite picks
fav_picks = model_top1[model_top1['IsFavourite']]
if len(fav_picks) > 0:
    wins = fav_picks['Winner'].sum()
    strike = wins / len(fav_picks) * 100
    profit = (fav_picks['Winner'] * fav_picks['StartingPrice']).sum() - len(fav_picks)
    roi = profit / len(fav_picks) * 100
    avg_odds = fav_picks['StartingPrice'].mean()
    progress(f"\nModel's Top Pick when IS favourite:")
    progress(f"  {len(fav_picks)} bets | {wins} wins | {strike:.1f}% strike | ${avg_odds:.2f} avg | ROI {roi:+.1f}%")

# ============================================================================
# STEP 7: Analyze by confidence level
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 7: Analyzing by model confidence...")

# High confidence = model probability much higher than implied probability
model_top1['Model_Edge'] = model_top1['Pred_Prob'] - model_top1['ImpliedProb']

for edge_threshold in [0.05, 0.10, 0.15, 0.20]:
    high_conf = model_top1[model_top1['Model_Edge'] >= edge_threshold]
    if len(high_conf) >= 20:
        wins = high_conf['Winner'].sum()
        strike = wins / len(high_conf) * 100
        profit = (high_conf['Winner'] * high_conf['StartingPrice']).sum() - len(high_conf)
        roi = profit / len(high_conf) * 100
        avg_odds = high_conf['StartingPrice'].mean()
        daily = len(high_conf) / 91  # 3 months
        progress(f"Model Edge >= {edge_threshold:.0%}: {len(high_conf)} bets ({daily:.1f}/day) | {strike:.1f}% | ${avg_odds:.2f} | ROI {roi:+.1f}%")

# ============================================================================
# STEP 8: Analyze by track tier
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 8: Analyzing by track tier...")

for tier in ['Metro', 'Provincial', 'Country']:
    tier_picks = model_top1[model_top1['TrackTier'] == tier]
    if len(tier_picks) >= 50:
        wins = tier_picks['Winner'].sum()
        strike = wins / len(tier_picks) * 100
        profit = (tier_picks['Winner'] * tier_picks['StartingPrice']).sum() - len(tier_picks)
        roi = profit / len(tier_picks) * 100
        avg_odds = tier_picks['StartingPrice'].mean()
        progress(f"{tier:12s}: {len(tier_picks)} bets | {strike:.1f}% strike | ${avg_odds:.2f} avg | ROI {roi:+.1f}%")

# ============================================================================
# STEP 9: Price range analysis
# ============================================================================
progress("\n" + "=" * 100)
progress("STEP 9: Analyzing by price range...")

for low, high in [(1.5, 2.5), (2.0, 3.0), (2.5, 4.0), (3.0, 5.0), (4.0, 8.0), (8.0, 20.0)]:
    price_picks = model_top1[(model_top1['StartingPrice'] >= low) & (model_top1['StartingPrice'] < high)]
    if len(price_picks) >= 30:
        wins = price_picks['Winner'].sum()
        strike = wins / len(price_picks) * 100
        profit = (price_picks['Winner'] * price_picks['StartingPrice']).sum() - len(price_picks)
        roi = profit / len(price_picks) * 100
        progress(f"${low:.1f}-${high:.1f}: {len(price_picks):4d} bets | {strike:5.1f}% strike | ROI {roi:+6.1f}%")

conn.close()

progress("\n" + "=" * 100)
progress("ANALYSIS COMPLETE")
progress("=" * 100)
