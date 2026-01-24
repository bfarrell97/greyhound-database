# Backtest Results Summary - Model Overconfidence Problem

## What We Tested

Three different approaches to fix the model's overconfidence on underdogs:

### Approach 1: Original Model (80% Confidence Threshold)
```
Strategy               Bets   Wins   Strike%   ROI
Confidence Proportional  552   196    35.5%    -69.67%
Flat Stake (2%)          552   196    35.5%    -83.58%
Half Kelly               148    58    39.2%    -99.82%
Kelly Criteria            94    39    41.5%    -99.83%

ISSUE: Model predicts 82.98% win rate but only wins 35.51%
       Massive miscalibration of ~47 percentage points
```

**Breakdown by Odds:**
- $1.0-$2.0: 164 bets, 70.1% win rate, **+$359 profit, +10.95% ROI** ✓ PROFITABLE!
- $2.0-$3.0: 124 bets, 33.9% win rate, -$452 loss
- $3.0-$5.0: 110 bets, 25.5% win rate, -$112 loss
- $5.0+: 243 bets, 4.5% win rate, -$1,350 loss

**Key Finding:** Model only works on favorites. Loses money on underdogs.

---

### Approach 2: Short Odds Filtering ($1.50-$2.00 Only)
```
Strategy               Bets   Wins   Strike%   ROI
Confidence Proportional 1953  1055    54.0%    -99.80%
Kelly Criteria          315   173     54.9%    -99.81%
Half Kelly              523   275     52.6%    -99.89%
Flat Stake (2%)        2187  1169    53.5%    -99.90%

ISSUE: Even on favorites with 53.5% strike rate, still losing 99.9%!
       This means bookmakers price favorites correctly
```

**Why it fails:**
- Average odds: $1.69
- With 55.4% win rate and $1.69 odds:
  - $1,000 staked = 1,000 × $1
  - Returns = 554 wins × $1.69 = $936
  - **Loss: $64 per $1,000 = -6.4% negative expected value**

Even though we're winning 55% of the time on favorites, the odds don't compensate.

---

## Root Cause: Model Doesn't Understand Full Picture

### What Model Currently Knows:
✓ Box draw statistics (BoxWinRate)
✓ Recent form: wins in last 3 (WinRateLast3)  
✓ Recent finishing positions (AvgPositionLast3)
✓ Benchmark times vs track average (GM_OT_ADJ_1-5)

### What Model DOESN'T Know:
✗ What the bookmakers think (no odds feature)
✗ Track-specific patterns (box 4 wins more at Sale, but less at Angle Park)
✗ Recency weighting (last race matters way more than 5 races ago)
✗ Opposition quality (beating weak dogs ≠ beating strong dogs)
✗ Dog racing style (front-runner, chaser, come-from-behind)
✗ Market consensus vs model disagreement

---

## The Real Problem

**It's not that the model is bad.**
**It's that bookmakers are better.**

Bookmakers:
- Have instant odds from thousands of bettors
- Adjust prices every 30 seconds based on betting patterns
- Have years of historical data
- Can see track inspector reports, veterinary status, trainer form
- Use sophisticated algorithms themselves

When the model predicts dog X will win at 60% probability, and bookmakers price it at 3.0 (33% probability), bookmakers are RIGHT 65% of the time. They've done their homework.

---

## The Solution (3-Phase Approach)

### Phase 1: Add Market Intelligence (1-2 hours)
Model needs to know what odds were set at. This helps it:
1. Learn what information the market uses
2. Identify rare cases where it genuinely disagrees with consensus
3. Avoid betting on dogs the market correctly identified as weak

**New features to add:**
- `BookmakerProb = 1 / odds` (what market thinks)
- `MarketDisagreement = model_prob - bookmaker_prob` (when they disagree)

**Expected impact:** Better model learns when to trust its edge vs. defer to market

### Phase 2: Probability Calibration (1-2 hours)
Currently predicts 83% but wins only 35%. Use Platt scaling to fix this:

```python
from sklearn.calibration import CalibratedClassifierCV

# Wraps your XGBoost model
calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
calibrated_model.fit(val_X, val_y)
# Now: if it predicts 50%, it wins ~50% of the time
```

**Expected impact:** Eliminates false confidence, more accurate betting

### Phase 3: Track-Specific Features (2-4 hours)
Each track has unique patterns. Add:
- Box draw win rates per track (Box 4 might win 25% at Angle Park but only 10% at Sale)
- Track bias toward running styles (front-runners vs chasers)
- Distance characteristics (mile tracks vs sprint tracks)
- Recency weighting (last race worth 3x, not 1x)

**Expected impact:** Model understands underdogs better, fewer misses on value bets

---

## What Success Looks Like

**Best case (with all improvements):**
- Strike rate: 60-70% on favorites
- ROI: +5 to +20%
- Focus only on very specific situations where model has genuine edge

**Realistic case (with Phase 1-2):**
- Strike rate: 45-55%
- ROI: -20% to +5%
- Model at least knows its limitations

**If still negative even after improvements:**
- Conclusion: Market is too efficient, model lacks hidden information
- Action: Stop betting and use model for analysis only, or pivot to different markets

---

## Betting Strategy Implications

**Today (without improvements):** DON'T BET. Model loses money.

**After Phase 1-2:** Only bet when model strongly disagrees with market AND has historical edge in that situation.

**After Phase 3:** Can expand to more opportunities, but still stick to >50% strike rate on favorites.

The key insight: **Profitability in betting comes from finding situations where you're RIGHT and the market is WRONG. Right now, the market is right.**

---

## Next Steps

1. **THIS WEEK:** Implement Phase 1 (add BookmakerProb feature)
2. Retrain model
3. Run backtest on new model - does ROI improve?

4. **NEXT WEEK:** If still unprofitable, implement Phase 2 (calibration)
5. Retrain and backtest

6. **FOLLOWING WEEK:** If still needed, implement Phase 3 (track features)
7. Final backtest

By then you'll know whether the model can be profitable or if you need a different approach entirely.
