"""
Quick demonstration: Add market confidence features to model
This shows what additional features could improve the model

To implement:
1. Add these features to your feature extraction in greyhound_ml_model.py
2. Retrain the model with these new features
3. Backtest to see if it improves
"""

# Example of features to add to feature extraction:

def _extract_additional_features(self, entry_dict, odds, box_win_rates):
    """
    Additional features that help model understand market vs model disagreement
    and track-specific patterns
    """
    features = {}
    
    # MARKET EFFICIENCY FEATURES
    # These help the model learn what the market knows
    bookmaker_implied_prob = 1.0 / odds
    features['BookmakerProb'] = bookmaker_implied_prob
    
    # Model should learn when it disagrees with market
    # (will be populated after prediction, for now just store odds-based value)
    features['OddsValue'] = odds
    
    # TRACK-SPECIFIC BOX DRAW FEATURES
    # Some boxes perform better at specific tracks
    box_performance_multipliers = {
        # Examples - calculate from historical data per track
        'Angle Park': {
            1: 1.15,  # Box 1 wins 15% more than baseline here
            2: 1.05,
            3: 0.95,
            4: 1.25,  # Box 4 is strong at Angle Park
            5: 0.85,
            6: 0.90,
            7: 0.85,
            8: 0.95,
        },
        'Sale': {
            1: 0.90,
            2: 1.10,
            3: 1.20,
            4: 1.35,  # Box 4 very strong
            5: 0.95,
            6: 0.85,
            7: 0.95,
            8: 0.90,
        },
        # ... etc for other tracks
    }
    
    features['BoxDrawMultiplier'] = 1.0  # Will be populated with real data
    
    # RECENCY WEIGHTING
    # Give more weight to recent form
    # If you have last 5 races: position_1, position_2, position_3, position_4, position_5
    # Weight them as: 5, 4, 3, 2, 1 instead of 1, 1, 1
    # This is in your AvgPositionLast3 calculation
    
    # RACING STYLE INDICATORS
    # Front-runner: Multiple wins from boxes 1-3
    # Chaser: Multiple wins from boxes 4-8
    # Come-from-behind: Consistent top-3 finishes starting from behind
    # Not currently in model - would need race replay data
    
    return features


# IMPLEMENTATION IN greyhound_ml_model.py:
# Add these lines to _extract_greyhound_features() method:

"""
# After calculating existing features (BoxWinRate, AvgPositionLast3, etc):

# ADD THIS:
# Market efficiency signal
bookmaker_prob = 1.0 / weight  # 'weight' parameter is actually odds in context
features['BookmakerProb'] = bookmaker_prob

# Track-specific box adjustment (example values - calculate from data)
track_box_multipliers = {
    'Angle Park': {1: 1.15, 2: 1.05, 3: 0.95, 4: 1.25, 5: 0.85, 6: 0.90, 7: 0.85, 8: 0.95},
    'Sale': {1: 0.90, 2: 1.10, 3: 1.20, 4: 1.35, 5: 0.95, 6: 0.85, 7: 0.95, 8: 0.90},
    # Add all other tracks...
}

if track_name in track_box_multipliers:
    box_mult = track_box_multipliers[track_name].get(box, 1.0)
    features['BoxDrawMultiplier'] = box_mult
else:
    features['BoxDrawMultiplier'] = 1.0

# Opposition strength (simplified - average rating of dogs in last race)
if len(last_5) > 0:
    last_race = last_5.iloc[0]
    # Would need to look up other dogs in that race to calculate this
    # For now, use finish position as proxy (lower is stronger opposition)
    features['LastOppositionStrength'] = last_race['Position']

# Recency weighted form (weight last race 3x, 2nd-last 2x, 3rd-last 1x)
positions_with_weights = []
for i, (_, race) in enumerate(last_3.iterrows()):
    weight = 3 - i  # First race (most recent) gets weight 3
    pos = pd.to_numeric(race['Position'], errors='coerce')
    if not pd.isna(pos):
        positions_with_weights.append(pos * weight)

if positions_with_weights:
    features['WeightedAvgPositionLast3'] = sum(positions_with_weights) / sum([3, 2, 1])
"""


# WHAT TO CHANGE IN YOUR TRAINING CODE:
"""
When training the model:

# Old feature columns
feature_columns = ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 
                   'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5']

# New feature columns (add these)
feature_columns = ['BoxWinRate', 'AvgPositionLast3', 'WinRateLast3', 
                   'GM_OT_ADJ_1', 'GM_OT_ADJ_2', 'GM_OT_ADJ_3', 'GM_OT_ADJ_4', 'GM_OT_ADJ_5',
                   'BookmakerProb',           # NEW: Market's assessment
                   'BoxDrawMultiplier',       # NEW: Track-specific box advantage
                   'WeightedAvgPositionLast3', # NEW: Recency-weighted form
                   'LastOppositionStrength']   # NEW: Quality of last opposition

# Then train normally - XGBoost will learn which features matter most
"""


# VALIDATION APPROACH:
"""
After adding features and retraining:

1. Use cross-validation to ensure the new features don't overfit
2. Apply Platt scaling to calibrate probabilities:
   
   from sklearn.calibration import CalibratedClassifierCV
   
   # After training model on full data:
   calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
   calibrated_model.fit(X_val, y_val)
   
   # Now use calibrated_model.predict_proba() instead of model.predict_proba()
   # This adjusts probabilities to match actual win rates

3. Backtest the calibrated model
4. Check if strike rate at different probability levels now matches predictions
5. If calibrated strike rate ≈ predicted probability, you've fixed the overconfidence!
"""
