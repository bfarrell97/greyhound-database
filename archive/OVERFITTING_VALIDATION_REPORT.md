# OVERFITTING VALIDATION REPORT
## Greyhound Racing ML Model - December 8, 2025

---

## EXECUTIVE SUMMARY

**Your concern about overfitting is VALID and shows good judgment.**

Based on the 4 backtesting techniques from the video you shared, your model has **NOT** been properly validated against overfitting. Before betting real money, you MUST implement proper validation.

---

## CURRENT MODEL STATUS

### Training Data
- **Period**: 2023-2024
- **Records**: 705,864 entries, 98,126 winners (13.9%)
- **Model**: XGBoost Classifier
  - max_depth: 6 (GOOD - not too deep)
  - n_estimators: 200
  - learning_rate: 0.1

### Prediction Target
- **Period**: December 2025
- **Gap**: ~1 year between training and prediction
- **Validation**: ❌ NONE

---

## OVERFITTING RISK ASSESSMENT

Using the 4 backtesting techniques from the video:

### 1. Parameter Sensitivity ❌ NOT TESTED
**Status**: No testing done
**Risk**: HIGH

You haven't verified if your model performance depends on specific "magic number" parameters.

**What you need to do**:
- Retrain with max_depth = 4, 5, 6, 7, 8
- Retrain with n_estimators = 100, 150, 200, 250
- Check if accuracy stays consistent (±5%)
- **Red flag**: If accuracy drops >10% with small parameter changes

### 2. Walk-Forward Optimization ❌ NOT IMPLEMENTED
**Status**: Model trained on ALL 2023-2024 data at once
**Risk**: CRITICAL

This is your **BIGGEST RISK**. The model may have "memorized" patterns that don't exist in 2025.

**What you need to do**:
```
Train on: Jan 2023 - Mar 2024 (15 months) → Test on: Apr 2024
Train on: Feb 2023 - Apr 2024 (15 months) → Test on: May 2024
Train on: Mar 2023 - May 2024 (15 months) → Test on: Jun 2024
...
Train on: Sep 2023 - Nov 2024 (15 months) → Test on: Dec 2024
```

**Success criteria**:
- Out-of-sample performance within 20% of in-sample
- Example: If in-sample accuracy = 35%, out-of-sample should be ≥ 28%

### 3. Stress Testing ❌ NOT TESTED
**Status**: No testing done
**Risk**: MEDIUM

You don't know how the model performs under different conditions.

**What you need to do**:
- **Odds slippage**: Assume odds are 10-20% worse - does ROI stay positive?
- **Execution delay**: 5-minute delay between prediction and bet
- **Track conditions**: Wet vs dry track performance
- **Field size**: 6-dog vs 8-dog races

### 4. Monte Carlo Simulations ❌ NOT IMPLEMENTED
**Status**: No risk assessment done
**Risk**: HIGH

You don't know the true risk range of your betting strategy.

**What happens in reality**:
- Backtest shows: "50 bets, win 20, profit $500"
- Monte Carlo might show: "In worst case, you lose 15 in a row"
- **Can your bankroll survive worst-case scenario?**

---

## VALIDATION ATTEMPT RESULTS

### November 2025 Data Available ✅
- **Days**: 30 days (Nov 1-30, 2025)
- **Races**: 4,180 races
- **Entries**: 29,650 greyhound entries
- **Winners**: 4,201 (14.17% - matches expected baseline)

### Backtest Attempt ❌ TECHNICAL LIMITATION

**Issue**: The model's `predict_upcoming_races()` function requires data in `UpcomingBettingRaces` tables. November historical data is in `Races/GreyhoundEntries` tables (different schema).

**Conclusion**: Cannot easily backtest on November data without significant code refactoring.

---

## CRITICAL RED FLAGS 🚩

1. **No 2025 Validation**
   - Model trained on 2023-2024
   - Predicting Dec 2025
   - **Zero testing on 2025 data**

2. **Data Gap**
   - 1 year between training and prediction
   - Market conditions may have changed
   - Greyhound performance patterns may have shifted

3. **No Live Testing**
   - No paper trading history
   - No tracking of actual vs predicted outcomes
   - **Betting real money would be gambling, not investing**

---

## RECOMMENDED ACTION PLAN

### ⚠️ DO NOT BET REAL MONEY UNTIL:

1. ✅ **Complete Phase 1: Paper Trading (2-4 weeks)**
   - Save daily predictions to log file
   - Compare to next-day actual results
   - Track: win rate, ROI, max drawdown
   - **Minimum 50 predictions before considering real money**

2. ✅ **Validation Criteria**
   - Win rate > 15% (vs 12.5% random baseline)
   - Positive ROI > 5%
   - Maximum losing streak < 10
   - Model probability calibration (50% predictions win ~50% of time)

3. ✅ **Start Small if Validation Passes**
   - Use 1/8 Kelly criterion for bet sizing
   - Start with minimum bet amounts
   - Track every bet for first month
   - Only scale up after sustained profitability

---

## PAPER TRADING IMPLEMENTATION

### Daily Workflow

**Each Evening**:
```bash
# Run predictions for tomorrow
python greyhound_ml_model.py

# Save predictions to log
python save_daily_predictions.py
```

**Next Day**:
```bash
# Compare predictions to actual results
python validate_predictions.py

# Update tracking metrics
python calculate_roi.py
```

### What to Track

Create a CSV log with:
- Date
- Greyhound name
- Track
- Race number
- Predicted probability
- Model price (fair odds)
- Actual market odds
- Edge percentage
- Bet size (simulated)
- Actual outcome (win/loss)
- Actual finish position
- Profit/loss

### Success Metrics (After 14+ Days)

**Minimum Requirements**:
- Total predictions: ≥50
- Win rate: ≥15% (at 50% confidence)
- ROI: ≥5%
- Longest losing streak: ≤10
- Sharpe ratio: >0.5 (if calculable)

**Good Performance**:
- Win rate: ≥20% (at 50% confidence)
- ROI: ≥10%
- Win rate: ≥30% (at 80% confidence)
- Consistent edge across different tracks

**Excellent Performance**:
- Win rate: ≥25% (at 50% confidence)
- ROI: ≥15%
- Win rate: ≥40% (at 80% confidence)
- Model probabilities well-calibrated

---

## RISK MANAGEMENT

### Bankroll Sizing (When/If You Go Live)

**Never risk more than 2% of bankroll on single bet**

Use fractional Kelly criterion:
```
Bet Size = (Edge × Probability / Odds) × Bankroll × (1/4 or 1/8)
```

Example:
- Bankroll: $1,000
- Model probability: 40%
- Market odds: $3.00 (33% implied)
- Edge: (0.40 - 0.33) / 0.33 = 21%
- Full Kelly: 21% of $1,000 = $210 (TOO RISKY)
- **1/4 Kelly: $52.50 bet** ← USE THIS
- **1/8 Kelly: $26.25 bet** ← EVEN SAFER

### Stop-Loss Rules

**Implement automatic stop-loss**:
- Stop if down 10% of bankroll in any week
- Stop if 10+ losing bets in a row
- Stop if actual win rate < 10% after 50 bets
- **Re-evaluate model if any stop-loss triggered**

---

## ALTERNATIVE: IMMEDIATE VALIDATION

If you want validation results THIS WEEK instead of waiting 2-4 weeks:

### Option: Backfill November Data

1. Copy November historical data into UpcomingBetting tables
2. Run model predictions on each day
3. Compare to actual results
4. Calculate win rate and ROI

**Estimated effort**: 4-8 hours of coding
**Benefit**: Immediate feedback on model performance

---

## FINAL RECOMMENDATION

### DO THIS IMMEDIATELY:
1. ✅ Start paper trading TODAY
2. ✅ Track predictions vs outcomes for 14+ days
3. ✅ Only bet real money if validation passes

### DO THIS SOON:
4. ⏰ Implement walk-forward validation
5. ⏰ Parameter sensitivity testing
6. ⏰ Monte Carlo risk simulation

### DON'T DO THIS:
❌ Bet real money without validation
❌ Use full Kelly criterion (too risky)
❌ Ignore losing streaks
❌ Retrain model daily (leads to overfitting)

---

## CONCLUSION

Your instinct to question overfitting is **spot-on**. Many gamblers lose money because they skip proper validation.

The video's 4 techniques are standard practice in quantitative trading firms for good reason - they separate luck from skill, and prevent costly mistakes.

**Bottom line**: Your model MIGHT be profitable, or it MIGHT be overfit. You won't know until you validate it. The only responsible path forward is paper trading for 2-4 weeks minimum.

**Good luck, and trade responsibly!**

---

*Report generated: December 8, 2025*
*Model version: greyhound_model.pkl*
*Training period: 2023-2024*
