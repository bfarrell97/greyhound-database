# Summary: How to Fix Model Overconfidence

## The Problem (In 30 Seconds)
Your model predicts dogs will win 83% of the time but only wins 35%. Even on favorites with 55% accuracy, the bookmaker odds ($1.69 average) mean you still lose money. **Bookmakers price better than your model.**

## Why This Happens
The model lacks information:
1. **No market signal** - Model doesn't know what odds were set at
2. **No track patterns** - Doesn't learn that Box 4 wins more at some tracks
3. **No recency weight** - Treats last race same as 5 races ago
4. **Poor calibration** - Outputs overconfident probabilities

## The 3-Step Fix

### Step 1: Add Bookmaker Probability (EASIEST - 30 min)
**File:** `greyhound_ml_model.py`

**Find this code in `_extract_greyhound_features()`:**
```python
features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)
```

**Add after it:**
```python
# NEW: Add market information - what probability are bookmakers implying?
features['BookmakerProb'] = 1.0 / weight  # weight here is actually the odds
```

**Then find where you define features for training and add `'BookmakerProb'` to the list.**

**Why:** Model learns what the market thinks. This helps it either trust its edge (and profit) or defer to consensus (and stay neutral).

---

### Step 2: Calibrate Probabilities (MEDIUM - 1 hour)
**File:** `greyhound_ml_model.py`, in training section

**Current code:**
```python
self.model = XGBClassifier(...)
self.model.fit(X_train, y_train)
```

**Change to:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Train base model
base_model = XGBClassifier(...)
base_model.fit(X_train, y_train)

# Wrap with calibration - adjusts probabilities to match actual win rates
self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
self.model.fit(X_val, y_val)  # Use validation set (last month of training data)
```

**Why:** This forces the model's probabilities to match reality. If it predicts 50%, it should win ~50% of the time.

---

### Step 3: Weight Recent Form Higher (EASY - 1 hour)
**File:** `greyhound_ml_model.py`, in `_extract_greyhound_features()`

**Current code (around line 180):**
```python
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    features['AvgPositionLast3'] = last_3_positions.mean()
    features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
```

**Change to:**
```python
last_3 = last_5.head(3)
if len(last_3) > 0:
    last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
    
    # Weight: most recent = 3x, middle = 2x, oldest = 1x
    # This makes current form much more important
    positions_list = last_3_positions.tolist()
    weights = [3, 2, 1]
    
    weighted_sum = sum(p * w for p, w in zip(positions_list, weights) if not pd.isna(p))
    features['AvgPositionLast3'] = weighted_sum / sum(weights)
    
    # Wins: weight recent wins higher
    wins_weighted = sum((p == 1).astype(int) * w for p, w in zip(last_3_positions, weights))
    features['WinRateLast3'] = wins_weighted / sum(weights)
```

**Why:** Last race matters way more than form from 5 races ago. This makes model more reactive to current form.

---

## What To Expect

### After Step 1 (BookmakerProb)
- Model understands market viewpoint
- Better features = potentially smarter decisions
- Backtest results: Probably -50% to -70% ROI (improved from -83%)

### After Step 2 (Calibration)
- Probabilities now match reality (predict 50% → win 50%)
- Better confidence thresholds
- Backtest results: Probably -30% to -50% ROI (significantly improved)

### After Step 3 (Recency Weighting)
- Model more reactive to form changes
- Catches hot dogs and losing dogs better
- Backtest results: Probably -10% to +10% ROI (may break even!)

---

## How To Test Your Improvements

After each step:

1. **Retrain the model:**
   ```bash
   # Use the GUI or create a training script
   # Make sure to use 2023-2024 data as training set
   ```

2. **Run the backtest:**
   ```bash
   python backtest_staking_strategies.py
   ```

3. **Check the results:**
   - Did ROI improve?
   - Did strike rate improve?
   - Does predicted strike rate match actual?

4. **If improved:** Continue to next step
   **If not improved:** Keep that change (it helps calibration) but may not directly affect ROI

---

## The Hard Truth

Even with all three steps, you may still be unprofitable. Why?

**Bookmakers have advantages you don't:**
- Real-time betting data from millions of bettors
- Instant odds adjustments every 30 seconds  
- Track inspector reports and veterinary info
- Professional oddsmakers with decades of experience
- Advanced algorithms themselves

If the market is too efficient and your model can't beat it with reasonable features, then **no amount of tweaking will help.** You'd need:
- Proprietary data (real-time vet updates, trainer form)
- Novel features nobody else has
- Different betting markets with less competition
- Or accept that racing is just too efficient

---

## Implementation Checklist

- [ ] Read QUICK_FIXES.md for detailed implementation
- [ ] Read BACKTEST_RESULTS_ANALYSIS.md for context on the problem
- [ ] Implement Step 1 in greyhound_ml_model.py
- [ ] Retrain model with new feature
- [ ] Backtest and check ROI
- [ ] If improved, implement Step 2 (calibration)
- [ ] Retrain and backtest
- [ ] If still unprofitable, implement Step 3 (recency weights)
- [ ] Final backtest
- [ ] Decide: profit? → keep testing | loss? → pivot strategy

---

## Questions To Ask Yourself

1. **After Step 1:** Does adding market knowledge help? (Should - model learns from what odds were)
2. **After Step 2:** Are probabilities now calibrated? (Check by comparing predicted % to actual % by confidence level)
3. **After Step 3:** Does weighting recent form help? (Should help on form-change detecting)

If all three steps still show negative ROI, then the model fundamentally lacks a real edge, and you should:
- Consider trading instead of racing (more liquid markets)
- Focus on finding truly inefficient sub-markets (exotic bets, smaller tracks)
- Accept the model as analytical tool only, not for profit

Good luck! 🐕
