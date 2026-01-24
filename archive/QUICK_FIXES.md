# Quick Action Items To Fix Model Overconfidence

## TL;DR - Do These 3 Things:

### 1. ADD BOOKMAKER PROBABILITY FEATURE (30 min)
The model should know what odds the dog was at. This tells it what the market thinks.

**Location:** `greyhound_ml_model.py`, method `_extract_greyhound_features()`

**Add this line after calculating other features:**
```python
features['BookmakerProb'] = 1.0 / odds  # odds parameter already exists
```

Then add `'BookmakerProb'` to your `feature_columns` list when training.

**Why:** Market prices contain valuable information. When model predicts 60% but odds are 3.0 (33%), that disagreement matters. XGBoost can learn patterns from it.

---

### 2. APPLY PROBABILITY CALIBRATION (1 hour)
Your model's probabilities are systematically wrong. Fix them mathematically.

**Location:** `greyhound_ml_model.py`, in training section

**Replace this:**
```python
# Old: just use raw model probabilities
self.model = XGBClassifier(...)
self.model.fit(X_train, y_train)
```

**With this:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Train base model first
base_model = XGBClassifier(...)
base_model.fit(X_train, y_train)

# Calibrate on validation set - this adjusts probability outputs
# to match actual win rates
self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
self.model.fit(X_val, y_val)  # Use validation set for calibration

# Save the calibrated model
```

**Why:** This mathematically forces the model's predictions to match actual outcomes. If it predicts 80%, it should win ~80% of the time (currently it wins only 35%).

---

### 3. WEIGHT RECENT RACES HIGHER (1 hour)
Current form matters more than form from 5 races ago.

**Location:** `greyhound_ml_model.py`, in `_extract_greyhound_features()`

**Current code (around line ~180):**
```python
# Recent form from last 3 races
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    features['AvgPositionLast3'] = last_3_positions.mean()  # ← EQUAL WEIGHTING
    features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
```

**Change to (weighted recency):**
```python
# Recent form from last 3 races - WEIGHTED by recency
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    
    # Weight: most recent race = 3x, 2nd most recent = 2x, 3rd most recent = 1x
    # This makes current form much more important
    weights = [3, 2, 1]  # weights for 1st, 2nd, 3rd most recent
    weighted_positions = []
    for pos, w in zip(last_3_positions, weights):
        if not pd.isna(pos):
            weighted_positions.append(pos * w)
    
    if weighted_positions:
        features['AvgPositionLast3'] = sum(weighted_positions) / sum(weights)
    
    # Wins: weight recent wins higher (if won last race = 3 points, etc)
    wins_weighted = sum((p == 1).astype(int) * w for p, w in zip(last_3_positions, weights))
    features['WinRateLast3'] = wins_weighted / sum(weights)
```

**Why:** A dog that won 3 races ago is less hot than one that just won. Weighting fixes this.

---

## Implementation Steps

**Step 1: Add BookmakerProb Feature (Easiest)**
- Open `greyhound_ml_model.py`
- Find `_extract_greyhound_features()` method
- Add one line: `features['BookmakerProb'] = 1.0 / odds`
- Find where you define `self.feature_columns` 
- Add `'BookmakerProb'` to the list
- Retrain model

**Step 2: Apply Calibration**
- Open `greyhound_ml_model.py`
- In the training section, add calibration wrapper
- This requires a validation set (use last month of training data)
- Retrain model

**Step 3: Weight Recent Form**
- Modify the `AvgPositionLast3` calculation as shown
- Only 3 lines of code to change
- Retrain model

---

## Expected Results After These Fixes

**Before fixes:**
- Strike rate: 35.5% (at 80% confidence)
- ROI: -69.7%
- Prediction calibration: Model predicts 83%, wins only 35%

**After fixes:**
- Strike rate: Probably 40-45% (better calibration = fewer false high confidence bets)
- ROI: Likely still negative, but **improved** to -40% to -50%
- Prediction calibration: Should match (predict 50%, win ~50%)

**Note:** Even after these fixes, you may still be unprofitable because bookmakers price efficiently. But at least:
1. You'll know your true edge (if any)
2. You won't overestimate your confidence
3. You can make informed decisions

---

## If Still Unprofitable After Step 1-3

Then you need Phase 2: **Track-specific features**
- Each track has unique patterns (some favor inside runners, some outside)
- Some tracks are "soft track" specialists
- Different distances favor different dog types

This requires calculating per-track statistics and adding them as features.

---

## Files to Edit

- `greyhound_ml_model.py` - Main model file where you make all changes
  - `_extract_greyhound_features()` - Add BookmakerProb, weight recent form
  - Training section - Add Calibration

That's it. No changes needed to backtest script or other files.
