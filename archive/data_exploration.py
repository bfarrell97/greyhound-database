"""
Data exploration: What variables are available and what correlates with wins on $1.50-$2.00?
This will help shape the next model improvements
"""

import sqlite3
import pandas as pd
import numpy as np
import sys

conn = sqlite3.connect('greyhound_racing.db')

print("="*80)
print("MODEL IMPROVEMENT RESEARCH")
print("="*80)

print("\n1. AVAILABLE VARIABLES IN DATABASE")
print("-"*80)

# Check what columns exist
tables = {
    'GreyhoundEntries': 'Individual race entries',
    'Greyhounds': 'Dog information',
    'Races': 'Race information',
    'RaceMeetings': 'Meeting/track information',
    'Tracks': 'Track details'
}

for table, desc in tables.items():
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"\n{table} ({desc}):")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

print("\n\n2. CURRENT FEATURE PERFORMANCE")
print("-"*80)

# Show what we're currently using
print("""
Current 8 Features (with recency weighting):
  1. BoxWinRate         - Win rate from this box at this track/distance
  2. AvgPositionLast3   - Average finish position (last 3 races)
  3. WinRateLast3       - Win rate (last 3 races)
  4-8. GM_OT_ADJ_1..5   - Combined benchmark adjustments (with 2x, 1.5x, 1, 1, 1 weights)

Current Performance on $1.50-$2.00:
  - 284 bets, 63.7% strike, +1.21% ROI

Target: Beat this with new variables or weights
""")

print("\n3. CANDIDATE NEW VARIABLES")
print("-"*80)

# Query to find what data is available
queries = {
    'Box Draw Distribution': """
        SELECT ge.Box, COUNT(*) as count, 
               SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM GreyhoundEntries ge
        WHERE ge.Box IS NOT NULL AND ge.Position IS NOT NULL
        GROUP BY ge.Box
        ORDER BY ge.Box
    """,
    
    'Distance Win Rates': """
        SELECT r.Distance, COUNT(*) as count,
               SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        WHERE ge.Position IS NOT NULL
        GROUP BY r.Distance
        ORDER BY r.Distance
    """,
    
    'Metro vs Non-Metro': """
        SELECT 
               CASE WHEN t.TrackName IN ('Wentworth Park', 'Albion Park', 'Angle Park', 'Hobart',
                                          'Launceston', 'Sandown Park', 'The Meadows', 'Cannington') 
                    THEN 'Metro' ELSE 'Non-Metro' END as track_type,
               COUNT(*) as count,
               SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        GROUP BY track_type
    """,
    
    'Weight Impact': """
        SELECT 
               CASE WHEN ge.Weight < 30 THEN 'Light (<30kg)'
                    WHEN ge.Weight BETWEEN 30 AND 33 THEN 'Medium (30-33kg)'
                    ELSE 'Heavy (33+kg)' END as weight_class,
               COUNT(*) as count,
               SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM GreyhoundEntries ge
        WHERE ge.Weight IS NOT NULL AND ge.Position IS NOT NULL
        GROUP BY weight_class
    """,
    
    'Price Effect on Strike Rate': """
        SELECT 
               CASE WHEN ge.StartingPrice <= 1.5 THEN '$1.00-$1.50'
                    WHEN ge.StartingPrice <= 2.0 THEN '$1.50-$2.00'
                    WHEN ge.StartingPrice <= 3.0 THEN '$2.00-$3.00'
                    WHEN ge.StartingPrice <= 5.0 THEN '$3.00-$5.00'
                    ELSE '$5.00+' END as odds_range,
               COUNT(*) as count,
               SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM GreyhoundEntries ge
        WHERE ge.StartingPrice IS NOT NULL AND ge.Position IS NOT NULL
        GROUP BY odds_range
        ORDER BY odds_range
    """
}

for name, query in queries.items():
    print(f"\n{name}:")
    print("-"*40)
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))

print("\n\n4. FEATURE IMPORTANCE FROM CURRENT MODEL")
print("-"*80)

import pickle
with open('greyhound_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
feature_cols = model_data['feature_columns']

if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance (top to bottom):")
    print(importance.to_string(index=False))
    
    # Calculate which GM_OT_ADJ are most important
    gm_features = importance[importance['Feature'].str.startswith('GM_OT_ADJ')]
    print("\nGM_OT_ADJ Breakdown (by race recency):")
    for idx, row in gm_features.iterrows():
        race_num = row['Feature'].split('_')[-1]
        print(f"  Race {race_num}: {row['Importance']*100:.2f}%")

print("\n\n5. PROPOSED IMPROVEMENTS TO TEST")
print("-"*80)

print("""
A) RECENCY WEIGHTING OPTIMIZATION
   Current: [2.0, 1.5, 1.0, 1.0, 1.0]
   Test alternatives:
   - Aggressive: [3.0, 1.5, 0.8, 0.5, 0.3] (heavily favor most recent)
   - Moderate:   [2.0, 1.5, 1.2, 0.9, 0.7] (gradual decay)
   - Balanced:   [2.5, 1.5, 1.0, 1.0, 1.0] (slightly more aggressive)

B) NEW FEATURES TO ADD
   Based on available data:
   - Weight class (light/medium/heavy) - may correlate with speed
   - Days since last race - fitness indicator
   - Win rate at specific distance - distance preference
   - Box draw effect - could matter for short odds
   - Recent form consistency - variance in results

C) TRACK WEIGHTING OPTIMIZATION
   Current: Metro 1.0, Provincial 0.7, Country 0.3
   Test:
   - City-specific: Wentworth/Albion/Angle 1.2, other Metro 0.9, Prov 0.7, Country 0.3
   - Flat: All 1.0 (test if weighting helps at all)
   - Inverse: Reverse weighting to test if country tracks are better

D) DISTANCE-SPECIFIC MODELS
   Build separate models for:
   - 400m races
   - 500m races  
   - 600m races
   - 700m+ races
   Test if edge varies by distance

E) ODDS-SPECIFIC FEATURES
   Since edge is on $1.50-$2.00:
   - Focus training only on this range
   - Different feature importance on short odds
   - May need different thresholds per odds band
""")

print("\n\n6. QUICK DECISION FRAMEWORK")
print("-"*80)

print("""
WHICH TO TEST FIRST? (in order of impact potential)

HIGH IMPACT (likely to improve 1-2% ROI):
  1. Recency weight optimization [easy, quick test]
  2. Add weight/distance features [medium effort]
  3. Odds-specific training [high effort, high potential]

MEDIUM IMPACT (likely 0.5-1% ROI):
  4. Track/city weighting tuning [easy test]
  5. Days since last race [easy feature add]

LOWER IMPACT:
  6. Box draw analysis [may be bookmaker-priced)
  7. Distance-specific models [complex, marginal gain)

RECOMMENDATION:
  Start with #1: Test 3-4 recency weight configs quickly
  Then #2: Add weight + distance features, retrain
  Then #3: If still not hitting target, try odds-specific training
""")

conn.close()

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
print("""
NEXT STEP: You pick what to test!

Which improvements interest you most?
1. Recency weight tuning (quickest)
2. New features (weight, days_since_race, distance_preference)
3. City track weighting 
4. Odds-specific training
5. Multiple tests in parallel

Let me know and I'll implement the tests!
""")
