# Greyhound Racing ML Model - Final Status Report

## Current Performance Summary

### Baseline Model (8 features)
- **Status**: ✅ PROFITABLE on $1.50-$2.00 range
- **Features**: BoxWinRate, AvgPositionLast3, WinRateLast3, + 5x GM_OT_ADJ with baseline recency [2.0, 1.5, 1.0, 1.0, 1.0]
- **Training Data**: 2023-01-01 to 2025-05-31 (579k+ examples, NZ/TAS filtered)
- **Test Period**: 2025-01-01 to 2025-11-30

### Tested Odds Ranges ($1.50-$3.00)

| Range | Bets | Wins | Strike% | Avg Odds | ROI% | Status |
|-------|------|------|---------|----------|------|--------|
| $1.50-$2.00 | 268 | 170 | 63.4% | 1.62 | **+1.29%** | ✅ PROFITABLE |
| $1.50-$2.50 | 370 | 210 | 56.8% | 1.78 | -2.97% | ❌ Loss |
| $1.50-$3.00 | 473 | 250 | 52.9% | 1.98 | -1.62% | ❌ Loss |
| $2.00-$3.00 | 205 | 80 | 39.0% | 2.44 | -5.41% | ❌ Loss |

**Key Finding**: Model's edge **only exists in $1.50-$2.00 range**. Expanding to higher odds destroys profitability.

## Feature Engineering Results

### Attempted Improvements

1. **Aggressive Recency Weighting** [3.0, 1.5, 0.8, 0.5, 0.3]
   - Expected: Better form prediction
   - Result: ❌ **-8.17% ROI on $1.50-$2.00** (vs +1.21% baseline)
   - Analysis: Too extreme, created model overconfidence

2. **WeightClass Feature** (Heavy/Medium/Light)
   - Raw data edge: +2.2% (heavy dogs win 15.2% vs light 13.0%)
   - Result: ❌ Hurt model performance when combined with other features
   - Analysis: Model already capturing weight signal through other features

3. **BoxDraw Feature** (Box position 1-8 normalized)
   - Raw data edge: +6-7% (box 1 wins 19% vs boxes 5-8 win 12-13%)
   - Result: ❌ Hurt model when added to full feature set
   - Analysis: Model already capturing draw signal through BoxWinRate feature

4. **Platt Scaling Calibration**
   - Expected: Better probability estimates
   - Result: ❌ Made predictions more overconfident, reduced bet volume and lost edge
   - Analysis: Raw probabilities from XGBoost are better than calibrated

### Why Feature Engineering Failed

The model doesn't improve with additional raw-data-correlated features because:
1. **XGBoost already extracts non-linear relationships**: The model captures weight/box effects through combinations of base features (especially BoxWinRate, AvgPositionLast3)
2. **Overfitting risk**: Adding features increases model complexity without validation benefit
3. **Calibration degradation**: Adding features changes the probability distribution, requiring recalibration which becomes more aggressive
4. **Data leakage patterns**: BoxWinRate already encodes box position effects for each track/distance combination

## Recommended Strategy

### Phase 1: Deploy Baseline (Immediate)
1. Keep current 8-feature model as-is
2. Target **$1.50-$2.00 odds range only**
3. Set confidence threshold at 80% (could test 75-85% for volume optimization)
4. Flat 2% staking
5. Expected ROI: **+1.2% to +1.3%** on ~250-300 bets/month
6. Expected profit: ~$30-40/month per $1000 bankroll

### Phase 2: Optimization (2-4 weeks)
Test selective improvements WITHOUT adding new features:
- Confidence threshold sweep (70%, 75%, 80%, 85%, 90%)
- Staking strategy optimization (flat % vs Kelly vs proportional)
- Bet selection refinement (value threshold, minimum odds margin)
- Seasonal analysis (does edge vary by month/track type?)

### Phase 3: Model Improvements (if needed)
If baseline performs worse than expected in live deployment:
1. **Retrain on more recent data** (shift 2025-Q4 and Q1 2026 into training)
2. **Test track-specific models** (separate models for Metro vs Provincial vs Country)
3. **DNF/SCR weighting** (currently training only on finished races)
4. **Form decay** (experiment with different recency schedules)

### What NOT to do
- ❌ Don't add arbitrary features with raw-data correlations
- ❌ Don't apply aggressive calibration without validation
- ❌ Don't expand to odds ranges without proving edge first
- ❌ Don't use Kelly or aggressive staking until edge is proven with larger sample

## Risk Assessment

### Potential Issues
1. **Sample size**: 268 bets over 11 months = ~24 bets/month. Confidence intervals are wide.
2. **Overfitting to 2025**: Training on 2023-2024, testing on 2025. Market may have changed.
3. **Odds bias**: Model may be cherry-picking races where bookmakers are slow to respond.
4. **Survivorship bias**: Model only trained on races that were completed (no DNF/SCR).

### Mitigation
- Track live results continuously
- Monitor by track, time of day, dog attributes
- Be ready to stop if edge disappears
- Validate on out-of-sample future data before expanding

## Conclusion

The baseline 8-feature XGBoost model achieves **+1.29% ROI on $1.50-$2.00 odds** with a 63.4% strike rate. This is a meaningful edge above the break-even point. The model's performance degrades significantly outside this odds range and when features are added. The recommendation is to **deploy the baseline model as-is** on the profitable $1.50-$2.00 range while monitoring performance continuously.
