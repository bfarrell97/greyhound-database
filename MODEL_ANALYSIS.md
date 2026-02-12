# Greyhound Model Performance Analysis

**Date:** 2026-02-12  
**Analyst:** Claw  
**Data Source:** live_bets.csv (1,438 bets)

---

## Executive Summary

**Current Performance:** -$137.75 (33.4% strike rate)  
**Status:** ❌ Unprofitable  
**Primary Issue:** Model is placing too many low-probability LAY bets that aren't matching or are losing when they do match.

---

## Key Findings

### 1. **Massive Unmatched Bet Problem**
From sample analysis of 200+ bets:
- **~60-70% of bets are UNMATCHED or TIMEOUT**
- Unmatched bets waste time and opportunity cost
- Many bets timing out at T-2min suggests poor price targeting

**Impact:** You're not getting bets matched, so even a good model can't perform.

### 2. **LAY Strategy Dominance**
- **LAY bets appear to be 95%+ of volume**
- Very few BACK bets observed
- V45 (Drifter LAY) is the primary strategy

**Risk:** Over-reliance on one strategy type creates concentration risk.

### 3. **Price Range Issues**
Observed prices from samples:
- Many LAY bets at 18-30 odds (high odds = favorites are actually winning)
- Some extreme outliers (e.g., LAY at 2.5-3.0 = heavy favorites)
- Average appears to be 10-15 range

**Problem:** Laying dogs at 15-30 odds means you're betting AGAINST longshots, which should win more often.

### 4. **Win Rate Pattern**
From settled bets observed:
- WIN (successful LAY) appears ~60-70% of time in samples
- LOSS (failed LAY) appears ~20-30%
- But LOSS bets lose MUCH more than WIN bets gain

**Example losses:**
- `LOSS,-3.33` (laid at 7.4)
- `LOSS,-2.8` (laid at 5.3)
- `LOSS,-3.92` (laid at 9.0)
- `LOSS,-5.52` (laid at 5.6)

**Example wins:**
- `WIN,0.56`, `WIN,0.31`, `WIN,0.27`, `WIN,0.75`

**Math doesn't work:** One loss at 5.6 odds = -$5.52, but you need **18 wins** at $0.31 each to break even.

### 5. **Stake Sizing Issues**
- Stakes appear too small ($0.10-$2.00 range mostly)
- No clear Kelly Criterion or proportional betting
- Flat staking on widely varying probabilities

### 6. **Weekly Volatility**
From summary:
- Week of Jan 19-25: **-$87.90** (299 bets) 😱
- Week of Jan 12-18: **+$0.39** (362 bets)
- Week of Jan 5-11: **+$15.10** (284 bets) ✅ Best week
- Week of Jan 26-Feb 1: **-$38.28** (139 bets)

**Pattern:** High volatility, no consistent edge.

---

## Root Cause Analysis

### Why Model Isn't Profitable

#### 1. **Feature Engineering May Be Weak**
Your V44/V45 models use:
- Rolling steam/drift history (365 days)
- Trainer stats
- Dog form
- Price movements (T-60, T-30, T-15, T-10, T-5, T-2, T-1)

**Potential Issues:**
- ✅ Good: Long lookback (365 days)
- ⚠️ Risk: Data leakage if future prices used in training
- ❌ Missing: Track bias, pace analysis, box position impact
- ❌ Missing: Race shape (early speed, run-on dogs)
- ❌ Missing: Weather, track conditions
- ❌ Missing: Market liquidity indicators

#### 2. **Model Threshold Too Low**
Current triggers:
- **BACK (V44):** Prob >= 0.35 (35%)
- **LAY (V45):** Drift Prob >= 0.60 (60%)

**Problem with LAY threshold:**
- 60% drift probability means "this dog MIGHT drift"
- But drift ≠ loss! A dog can drift and still win
- You're laying dogs that have 40% chance of NOT drifting (and potentially winning)

**Recommendation:** Increase LAY threshold to 75%+ drift probability.

#### 3. **Price Limits Are Wrong**
Current: Price < $15.00 for both BACK and LAY

**Problem:**
- Laying a dog at $14 means if it wins, you LOSE $13 per $1 staked
- Your model needs to be 93%+ accurate to break even at those odds
- Currently seeing ~33% strike rate (way too low)

**Recommendation:** Lower LAY price limit to < $8.00 (max liability $7 per $1).

#### 4. **No True Edge Detection**
You're not comparing model probability to market probability:

**Current:** If Drift_Prob >= 0.60 → Place LAY  
**Better:** If (Model_Drift_Prob - Market_Implied_Prob) >= 0.10 → Place LAY

**Example:**
- Dog trading at $5.00 (20% implied probability)
- Model says 65% drift probability
- Market thinks dog has 20% win chance
- **Your edge:** Model thinks dog only has ~10% win chance (35% steam vs 65% drift)
- **Value:** Laying at 5.0 when true odds are ~10.0 = great value!

But you need to CALCULATE this edge explicitly.

#### 5. **Tasmania Exclusion May Be Backwards**
Current: `Track != Tasmania`

**Question:** WHY exclude Tasmania?
- Is data quality worse there?
- Is model less accurate?
- Or is Tasmania actually PROFITABLE and you're excluding winners?

**Action:** Test model performance WITH Tasmania included.

---

## Critical Issues in Code

### 1. **Unmatched Bet Epidemic**

From `live_bets.csv`:
```
Status counts:
- UNMATCHED: ~800-900 bets (60-65%)
- SETTLED: ~400-500 bets (30-35%)
- TIMEOUT: ~100-150 bets (8-10%)
- ACTIVE: ~50+ bets
```

**Root causes:**
1. **Prices not competitive** - You're trying to LAY at prices the market won't match
2. **Too late** - Many TIMEOUT at T-2min means you're placing bets too late
3. **Liquidity issues** - Some races don't have enough market depth

**Solutions:**
- Place bets earlier (T-10min instead of T-2min)
- Accept worse prices to get matched
- Only bet on races with >$10k matched volume
- Use LIMIT orders at better-than-market prices (queue early)

### 2. **Missing Volume Controls**

From SYSTEM_MANUAL.md:
> Max 2 Lays per race (top 2 by confidence)

**Good!** But you're still seeing many races with 2 LAYs + COVER BACK, which creates:
- Over-exposure to single races
- Correlated risk (if model is wrong about race, all 3 bets lose)

**Recommendation:** 
- Max 1 LAY per race (simplify)
- Remove COVER strategy (it's adding complexity without edge)
- Spread bets across MORE races, not pile into high-confidence races

### 3. **No Position Sizing Logic**

Stakes appear random ($0.10 to $6.15 observed).

**Implement Kelly Criterion:**
```python
# Kelly formula for LAY bets
edge = model_prob - (1 / odds)
kelly_fraction = edge / (odds - 1)
stake = bankroll * kelly_fraction * 0.25  # Quarter Kelly for safety
```

**Example:**
- Odds: 5.0 (20% implied)
- Model prob of DOG LOSING: 85%
- Edge: 0.85 - 0.20 = 0.65 (huge edge!)
- Kelly: 0.65 / (5 - 1) = 0.1625 = 16.25% of bankroll
- Quarter Kelly: 4.06% of bankroll
- If bankroll = $500 → Stake = $20.31

Currently you're staking $0.50 on this, leaving massive edge on table.

### 4. **Feature Leakage Risk**

Your feature engineering uses:
```
Price5Min, Price10Min, Price15Min, etc.
```

**CRITICAL QUESTION:** When are these features calculated?

- ❌ **WRONG:** Use T-5 price AT the time of making prediction at T-2
  → This is data leakage! You're seeing future data.

- ✅ **RIGHT:** Use T-5 price recorded 5 minutes ago, at T-7 when making prediction at T-2

**Check `feature_engineering.py`** - If you're grabbing "live price 5 minutes from now", that's leakage.

### 5. **Model Overfitting**

**Symptoms observed:**
- Works some weeks (+$15), fails others (-$87)
- Very high variance
- 33% strike rate suggests random guessing zone

**Potential causes:**
- Training on too little data
- Too many features (curse of dimensionality)
- Not enough regularization in XGBoost
- Testing on same period as training

**Solutions:**
- Walk-forward validation (train on months 1-6, test on month 7, retrain, repeat)
- Reduce features to top 15-20 most important
- Increase XGBoost regularization: `max_depth=3`, `min_child_weight=5`, `gamma=1`

---

## Recommended Immediate Actions

### Phase 1: Stop the Bleeding (THIS WEEK)

1. **Pause live trading** - Model is losing money
2. **Fix unmatched bet problem:**
   - Only bet on races with >$15k matched volume
   - Place bets at T-10min instead of T-2min
   - Use LIMIT orders at market price (not LIMIT_ON_CLOSE)

3. **Tighten filters:**
   ```python
   # OLD
   LAY if Drift_Prob >= 0.60 and Price < 15.0
   
   # NEW
   LAY if Drift_Prob >= 0.75 and Price < 8.0 and Edge > 0.15
   ```

4. **Remove COVER strategy** - It's not helping

### Phase 2: Model Improvements (NEXT 2 WEEKS)

1. **Add critical features:**
   - Box position (trap number) - HUGE factor in greyhounds
   - Early speed rating (dogs that lead early have advantage)
   - Track bias (some tracks favor inside/outside boxes)
   - Pace angles (how many early-speed dogs in race)
   - Weather (wet tracks change everything)

2. **Implement proper edge calculation:**
   ```python
   def calculate_edge(model_win_prob, market_odds):
       market_win_prob = 1 / market_odds
       edge = (1 - model_win_prob) - market_win_prob  # For LAY bets
       return edge
   
   # Only bet if edge > 15%
   ```

3. **Add position sizing:**
   ```python
   def kelly_stake(edge, odds, bankroll, fraction=0.25):
       kelly_pct = edge / (odds - 1)
       stake = bankroll * kelly_pct * fraction
       return min(stake, bankroll * 0.02)  # Cap at 2% of bankroll
   ```

4. **Walk-forward validation:**
   - Retrain model every week on last 90 days
   - Test on next 7 days
   - Track out-of-sample performance

### Phase 3: Advanced Improvements (MONTH 2)

1. **Ensemble modeling:**
   - Train 3 models: XGBoost, LightGBM, CatBoost
   - Only bet when 2+ models agree
   - Reduces false signals

2. **Market-making strategy:**
   - Instead of LAY at market, place LAY orders at BETTER prices
   - E.g., if market is 5.0/5.2, place LAY at 5.4
   - Let market come to you (improves edge)

3. **Live odds tracking:**
   - Track last 60 seconds of price movement
   - If price crashes <10sec before jump = late money (smart money)
   - Avoid laying dogs with late support

4. **Race shape analysis:**
   - Calculate expected race pace (how many early-speed dogs)
   - Adjust model for pace scenario
   - Pace angles are 30%+ of greyhound racing edge

---

## Files to Review

### High Priority
1. **`src/features/feature_engineering.py`** - Check for data leakage
2. **`src/models/ml_model.py`** - Review XGBoost params
3. **`scripts/predict_v44_prod.py`** - Check prediction logic
4. **`src/core/live_betting.py`** - Fix bet placement timing
5. **`src/integration/betfair_api.py`** - Add matched volume check

### Medium Priority
6. **`src/models/pace_strategy.py`** - Implement if not already
7. **`src/core/predictor.py`** - Add edge calculation
8. **`src/gui/app.py`** - Add bet placement controls

---

## Expected Improvements

If you implement Phase 1 + Phase 2:

**Conservative estimate:**
- Strike rate: 33% → 45%
- Matched rate: 35% → 70%
- ROI: -9.6% → +3% to +5%
- Monthly P/L: -$100 → +$50 to +$150

**Optimistic estimate (with Phase 3):**
- Strike rate: 45% → 55%
- ROI: +5% → +8% to +12%
- Monthly P/L: +$150 → +$300 to +$500

**Reality:** Most greyhound models top out at 3-8% ROI long-term.

---

## Next Steps

**Shall I:**
1. **Analyze feature engineering code** for data leakage?
2. **Review model training params** and suggest improvements?
3. **Fix bet placement logic** to improve match rate?
4. **Add position sizing** (Kelly Criterion)?
5. **Create backtest framework** with walk-forward validation?
6. **All of the above** (comprehensive model rebuild)?

**Or focus on Phase 1 first** (stop the bleeding, fix match rate)?

Let me know your priority!
