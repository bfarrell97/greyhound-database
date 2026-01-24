"""
Compare unified model vs track-specific models on $1.50-$3.00 odds
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import sys

# Load models
print("Loading models...")
with open('greyhound_model.pkl', 'rb') as f:
    unified_model_data = pickle.load(f)

with open('greyhound_model_metro.pkl', 'rb') as f:
    metro_model = pickle.load(f)

with open('greyhound_model_provincial.pkl', 'rb') as f:
    provincial_model = pickle.load(f)

with open('greyhound_model_country.pkl', 'rb') as f:
    country_model = pickle.load(f)

conn = sqlite3.connect('greyhound_racing.db')

# Track tier definitions
METRO_TRACKS = {
    'Wentworth Park', 'Albion Park', 'Angle Park', 'Hobart',
    'Launceston', 'Sandown Park', 'The Meadows', 'Cannington'
}

PROVINCIAL_TRACKS = {
    'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli',
    'Dapto', 'Maitland', 'Goulburn', 'Ipswich', 'Q Straight',
    'Q1 Lakeside', 'Q2 Parklands', 'Gawler', 'Devonport', 'Ballarat',
    'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'
}

def get_track_tier(track_name):
    if track_name in METRO_TRACKS:
        return 'metro'
    elif track_name in PROVINCIAL_TRACKS:
        return 'provincial'
    else:
        return 'country'

def get_track_weight(track_name, weights):
    tier = get_track_tier(track_name)
    return weights.get(tier, 0.3)

# Load test data
print("Loading test data...")
query = """
SELECT
    ge.EntryID, g.GreyhoundID,
    rm.MeetingDate, t.TrackName, r.Distance,
    ge.Box, ge.StartingPrice, ge.Position
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= '2025-01-01'
  AND rm.MeetingDate <= '2025-11-30'
  AND ge.StartingPrice > 1.5 AND ge.StartingPrice <= 3.0
  AND ge.Position IS NOT NULL
ORDER BY rm.MeetingDate, r.RaceNumber
"""

races = pd.read_sql_query(query, conn)
print(f"Loaded {len(races)} race entries in $1.50-$3.00 range\n")

feature_columns = [
    'BoxWinRate', 'AvgPositionLast3', 'WinRateLast3',
    'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5'
]

TRACK_WEIGHTS = {'metro': 1.0, 'provincial': 0.7, 'country': 0.3}

def get_track_tier_weight(track_name):
    if track_name in METRO_TRACKS:
        return TRACK_WEIGHTS['metro']
    elif track_name in PROVINCIAL_TRACKS:
        return TRACK_WEIGHTS['provincial']
    else:
        return TRACK_WEIGHTS['country']

def extract_features_for_race(greyhound_id, race_date, track_name, distance, box):
    """Extract features for a single race"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT t.TrackName, ge.Position,
               ge.FinishTimeBenchmarkLengths, rm.MeetingAvgBenchmarkLengths
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.GreyhoundID = ?
          AND rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
        ORDER BY rm.MeetingDate DESC LIMIT 5
    """, (greyhound_id, race_date))
    
    historical_races = cursor.fetchall()
    
    if len(historical_races) < 5:
        return None
    
    features = {}
    features['BoxWinRate'] = 0.125
    
    # Recent form
    last_3_positions = [int(row[1]) for row in historical_races[:3]]
    features['AvgPositionLast3'] = sum(last_3_positions) / len(last_3_positions) if last_3_positions else 4.5
    features['WinRateLast3'] = sum(1 for p in last_3_positions if p == 1) / len(last_3_positions) if last_3_positions else 0
    
    # GM_OT_ADJ with recency weighting
    recency_weights = [2.0, 1.5, 1.0, 1.0, 1.0]
    for i, row in enumerate(historical_races, 1):
        track, position, g_ot, m_ot = row
        g_val = float(g_ot) if g_ot else 0.0
        m_val = float(m_ot) if m_ot else 0.0
        
        track_weight = get_track_tier_weight(track)
        recency_weight = recency_weights[i - 1]
        features[f'GM_OT_ADJ_{i}'] = (g_val + m_val) * track_weight * recency_weight
    
    return features

# Extract features and get predictions
print("Extracting features and getting predictions...")
results = []

for idx, row in races.iterrows():
    if idx % 30000 == 0:
        print(f"  {idx}/{len(races)}")
        sys.stdout.flush()
    
    features = extract_features_for_race(
        row['GreyhoundID'], row['MeetingDate'],
        row['TrackName'], row['Distance'], row['Box']
    )
    
    if features is None:
        continue
    
    # Get feature vector
    X = pd.DataFrame([features])[feature_columns]
    
    # Get predictions from all models
    unified_pred = unified_model_data['model'].predict_proba(X)[0][1]
    
    tier = get_track_tier(row['TrackName'])
    if tier == 'metro':
        specific_pred = metro_model.predict_proba(X)[0][1]
    elif tier == 'provincial':
        specific_pred = provincial_model.predict_proba(X)[0][1]
    else:
        specific_pred = country_model.predict_proba(X)[0][1]
    
    is_winner = 1 if row['Position'] == '1' or row['Position'] == 1 else 0
    
    results.append({
        'Unified_Pred': unified_pred,
        'Specific_Pred': specific_pred,
        'Odds': row['StartingPrice'],
        'IsWinner': is_winner,
        'TrackTier': tier
    })

results_df = pd.DataFrame(results)
print(f"\nGot predictions for {len(results_df)} races\n")

# Compare at 80% confidence
print("="*70)
print("COMPARISON AT 80% CONFIDENCE THRESHOLD")
print("="*70)

for model_name in ['Unified', 'Specific']:
    pred_col = f'{model_name}_Pred'
    
    filtered = results_df[results_df[pred_col] >= 0.8]
    
    if len(filtered) == 0:
        print(f"\n{model_name} Model: No predictions at 80%+")
        continue
    
    wins = filtered['IsWinner'].sum()
    strike_rate = wins / len(filtered) * 100
    
    total_returned = (filtered['IsWinner'] * filtered['Odds']).sum() + (1 - filtered['IsWinner']).sum()
    total_staked = len(filtered)
    profit = total_returned - total_staked
    roi = (profit / total_staked) * 100
    
    print(f"\n{model_name} Model:")
    print(f"  Bets: {len(filtered)}")
    print(f"  Wins: {wins}")
    print(f"  Strike Rate: {strike_rate:.1f}%")
    print(f"  ROI: {roi:.2f}%")
    
    # Breakdown by track tier
    print(f"  By track tier:")
    for tier in ['metro', 'provincial', 'country']:
        tier_data = filtered[filtered['TrackTier'] == tier]
        if len(tier_data) > 0:
            tier_wins = tier_data['IsWinner'].sum()
            tier_sr = tier_wins / len(tier_data) * 100
            tier_returned = (tier_data['IsWinner'] * tier_data['Odds']).sum() + (1 - tier_data['IsWinner']).sum()
            tier_profit = tier_returned - len(tier_data)
            tier_roi = (tier_profit / len(tier_data)) * 100
            print(f"    {tier.title()}: {len(tier_data)} bets, {tier_sr:.1f}% strike, {tier_roi:.2f}% ROI")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ Recency weighting: +1.21% ROI on $1.50-$2.00")
print("✓ Track-specific models: Now testing if they improve $1.50-$3.00")
print("\nNext: If track-specific models help, combine with recency weighting")

conn.close()
