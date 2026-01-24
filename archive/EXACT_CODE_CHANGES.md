# Exact Code Changes For Fixing Model Overconfidence

## File: greyhound_ml_model.py

### Change #1: Add BookmakerProb Feature
**Location:** In `_extract_greyhound_features()` method, after line where BoxWinRate is set

**BEFORE:**
```python
# Box win rate
box_key = (track_id, distance, box)
features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)

# Recent form from last 3 races
last_3 = last_5.head(3)
```

**AFTER:**
```python
# Box win rate
box_key = (track_id, track_id, box)
features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)

# Market probability from odds (NEW)
features['BookmakerProb'] = 1.0 / weight if weight else 0.5

# Recent form from last 3 races
last_3 = last_5.head(3)
```

**Note:** The `weight` parameter is actually the odds (confusing naming). So `1.0 / weight` = implied probability from bookmaker odds.

---

### Change #2: Weight Recent Races Higher  
**Location:** In `_extract_greyhound_features()` method, the AvgPositionLast3 and WinRateLast3 calculation

**BEFORE:**
```python
# Recent form from last 3 races
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    features['AvgPositionLast3'] = last_3_positions.mean()
    features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
else:
    features['AvgPositionLast3'] = 4.5
    features['WinRateLast3'] = 0
```

**AFTER:**
```python
# Recent form from last 3 races - WEIGHTED by recency
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    
    # NEW: Weight recent races higher
    # Most recent race gets 3x weight, middle gets 2x, oldest gets 1x
    # This makes current form much more important than old form
    weights = [3, 2, 1]  # weights for most recent to oldest
    
    # Calculate weighted average position
    weighted_positions = []
    for pos, weight_val in zip(last_3_positions, weights):
        if not pd.isna(pos):
            weighted_positions.append(pos * weight_val)
    
    if weighted_positions:
        features['AvgPositionLast3'] = sum(weighted_positions) / sum(weights)
    else:
        features['AvgPositionLast3'] = 4.5
    
    # Calculate weighted win rate
    wins_weighted = 0
    for pos, weight_val in zip(last_3_positions, weights):
        if not pd.isna(pos) and pos == 1:
            wins_weighted += weight_val
    features['WinRateLast3'] = wins_weighted / sum(weights)
else:
    features['AvgPositionLast3'] = 4.5
    features['WinRateLast3'] = 0
```

---

### Change #3: Add BookmakerProb to Feature Columns
**Location:** Find where `self.feature_columns` is defined (usually in `__init__` or training method)

**BEFORE:**
```python
self.feature_columns = [
    'BoxWinRate',
    'AvgPositionLast3',
    'WinRateLast3',
    'GM_OT_ADJ_1',
    'GM_OT_ADJ_2',
    'GM_OT_ADJ_3',
    'GM_OT_ADJ_4',
    'GM_OT_ADJ_5'
]
```

**AFTER:**
```python
self.feature_columns = [
    'BoxWinRate',
    'AvgPositionLast3',
    'WinRateLast3',
    'GM_OT_ADJ_1',
    'GM_OT_ADJ_2',
    'GM_OT_ADJ_3',
    'GM_OT_ADJ_4',
    'GM_OT_ADJ_5',
    'BookmakerProb'  # NEW: Market's assessment of win probability
]
```

---

### Change #4: Apply Probability Calibration
**Location:** In training section, where XGBoost model is created

**BEFORE:**
```python
# Train model
self.model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

self.model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)
```

**AFTER:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Train base model
base_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

base_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)

# Calibrate probabilities to match actual outcomes
# This fixes the overconfidence problem
print("Calibrating model probabilities...")
self.model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Use sigmoid method for calibration
    cv=5  # 5-fold cross-validation
)
self.model.fit(X_val, y_val)  # Use validation set for calibration

print("Model calibrated successfully")
```

---

## Testing The Changes

### Test 1: Verify BookmakerProb Feature Is Captured
```python
# In your training data exploration code, after feature extraction:
print("Sample feature values:")
print(X.head())
# Should see 'BookmakerProb' column with values like 0.33, 0.5, 0.25, etc.
```

### Test 2: Verify Recency Weighting Works
```python
# Check that recent form is weighted higher
# Dog with position [1, 3, 5] (recent to old):
# Unweighted: (1+3+5)/3 = 3.0
# Weighted:   (1*3 + 3*2 + 5*1) / 6 = (3+6+5)/6 = 2.33
# So weighted version gives more credit to the recent win

# If you see numbers that match this pattern, recency weighting is working
```

### Test 3: Check Calibration
```python
# After training with calibration, test on validation set:
val_probs = self.model.predict_proba(X_val)[:, 1]

# Group by probability bins and check actual win rates
for prob_bin in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = (val_probs >= prob_bin - 0.05) & (val_probs < prob_bin + 0.05)
    if mask.sum() > 0:
        actual_win_rate = y_val[mask].mean()
        print(f"Predicted {prob_bin:.0%}: Actual win rate = {actual_win_rate:.0%}")

# If calibrated correctly, predicted should match actual
# Before: Predicted 80%, Actual 35% (BAD)
# After:  Predicted 50%, Actual 50% (GOOD)
```

---

## Order To Implement

1. **First:** Change #1 (Add BookmakerProb) + Change #3 (add to feature_columns)
   - Retrain and test
   
2. **Then:** Change #2 (Weight recent races)
   - Retrain and test

3. **Finally:** Change #4 (Apply calibration)
   - Retrain and test

Do them one at a time so you can see the impact of each change.

---

## Common Mistakes To Avoid

❌ **WRONG:** Using column index instead of feature name
```python
# DON'T do this:
X[0] = 1.0 / odds
```

✅ **RIGHT:** Use feature dictionary and DataFrame
```python
# DO this:
features['BookmakerProb'] = 1.0 / odds
```

---

❌ **WRONG:** Forgetting to add feature to feature_columns list
```python
# Your code extracts BookmakerProb but...
# It's not in feature_columns so model never uses it
```

✅ **RIGHT:** Add to both places
```python
features['BookmakerProb'] = ...  # Extract it
self.feature_columns = [..., 'BookmakerProb']  # Tell model to use it
```

---

❌ **WRONG:** Applying calibration on training data
```python
# This overfits calibration to training set
self.model = CalibratedClassifierCV(base_model, cv=5)
self.model.fit(X_train, y_train)
```

✅ **RIGHT:** Apply calibration on validation set
```python
# Use held-out data so calibration generalizes
self.model = CalibratedClassifierCV(base_model, cv=5)
self.model.fit(X_val, y_val)
```

---

## Questions?

If you run into issues:

1. **BookmakerProb is NaN:** Check that `weight` parameter isn't 0
2. **Weighted average is wrong:** Debug by printing intermediate values
3. **Model doesn't use new feature:** Verify it's in `feature_columns` list
4. **Calibration fails:** Make sure X_val and y_val have enough samples (>500 rows)

Good luck! 🐕
