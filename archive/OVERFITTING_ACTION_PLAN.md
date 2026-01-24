# OVERFITTING MITIGATION PLAN

## CRITICAL: Before Betting Real Money

### Phase 1: Validate on Recent Data (MUST DO FIRST)
1. **Scrape last 30 days of historical results** (Nov 8 - Dec 7, 2025)
   - Run: `populate_historical_data.py` for November 2025
   - Get ACTUAL race outcomes

2. **Paper trade backtest**
   - Make predictions for each day Nov 8 - Dec 7
   - Compare predictions to actual results
   - Calculate: Win rate, ROI, max drawdown

3. **Success criteria**:
   - At 50% confidence: Win rate > 20% (vs 12.5% baseline)
   - At 80% confidence: Win rate > 30%
   - Positive ROI after simulated kelly betting

4. **If backtest fails**:
   - STOP - Model is overfit to 2023-2024
   - Need to retrain with walk-forward validation

### Phase 2: Implement Walk-Forward Validation

Instead of training on ALL 2023-2024, do this:

```python
# Walk-forward training schedule:
Train: Jan 2023 - Mar 2024 (15 months) → Test: Apr 2024 (1 month)
Train: Feb 2023 - Apr 2024 (15 months) → Test: May 2024 (1 month)
Train: Mar 2023 - May 2024 (15 months) → Test: Jun 2024 (1 month)
...
Train: Sep 2023 - Nov 2024 (15 months) → Test: Dec 2024 (1 month)
```

**Success criteria**:
- Out-of-sample performance within 20% of in-sample
- Example: If in-sample accuracy = 35%, out-of-sample should be >= 28%

### Phase 3: Parameter Sensitivity Testing

Test model with different hyperparameters:

| max_depth | n_estimators | learning_rate | Expected Accuracy Range |
|-----------|--------------|---------------|-------------------------|
| 4         | 150          | 0.05          | ?                       |
| 5         | 175          | 0.08          | ?                       |
| 6         | 200          | 0.10          | ? (current)             |
| 7         | 225          | 0.12          | ?                       |
| 8         | 250          | 0.15          | ?                       |

**Success criteria**:
- Accuracy should vary by < 5% across parameter ranges
- No single "magic number" that makes model work

### Phase 4: Stress Testing

1. **Odds slippage test**:
   - Assume odds are 10% worse than shown (favorites shorter, longshots longer)
   - Does model still show positive ROI?

2. **Execution delay test**:
   - Assume 5-minute delay between prediction and bet
   - Simulate odds movement
   - Does model still profit?

3. **Track condition test**:
   - Filter to only wet track races
   - Does model still work?
   - Filter to only dry track races
   - Compare performance

4. **Field size test**:
   - 6-dog fields vs 8-dog fields
   - Does accuracy change dramatically?

### Phase 5: Monte Carlo Simulation

1. Take your backtested bets (from Phase 1)
2. Shuffle the order 1000 times
3. Generate 1000 different equity curves
4. Find worst-case scenario:
   - Longest losing streak
   - Maximum drawdown
   - Time to recovery

**Success criteria**:
- Can you psychologically handle worst-case losing streak?
- Can your bankroll survive worst-case drawdown?
- If worst-case loses entire bankroll, REDUCE bet sizing

## RED FLAGS That Mean Stop Betting

1. ❌ Out-of-sample performance << in-sample (difference > 30%)
2. ❌ Model only works with exact parameter values
3. ❌ Can't survive 10% odds slippage
4. ❌ Monte Carlo shows 50%+ drawdown risk
5. ❌ Win rate on recent data (Nov 2025) < 15%
6. ❌ Negative ROI on paper trading

## GREEN LIGHTS to Start Small Betting

1. ✅ Out-of-sample performance within 20% of in-sample
2. ✅ Model robust across parameter ranges (±3% accuracy variation)
3. ✅ Positive ROI even with 10% odds slippage
4. ✅ Monte Carlo shows max 25% drawdown
5. ✅ Win rate on recent data > 20% (at 50% confidence)
6. ✅ 2+ weeks of profitable paper trading

## Ongoing Monitoring (After Going Live)

### Weekly checks:
- Actual win rate vs predicted
- Actual ROI vs expected
- Track: every bet, every outcome, every edge

### Monthly checks:
- Rolling 30-day ROI
- Compare to Monte Carlo simulations
- Are results within expected range?

### Quarterly retraining:
- Only retrain if new data is significantly different
- Always keep a holdout set model never sees
- Re-run all validation tests

## Kelly Criterion Bankroll Management

Even with a good model, use fractional Kelly:

- Full Kelly: bet_size = (edge × probability) / odds
- **Use 1/4 Kelly or 1/8 Kelly** to reduce variance

Example:
- Model says 40% win probability
- Odds are $3.00 (33% implied)
- Edge = (0.40 - 0.33) / 0.33 = 21%
- Full Kelly = 21% of bankroll
- **1/4 Kelly = 5.25% of bankroll** ← USE THIS

## Timeline

### Week 1-2: Data Collection & Validation
- Scrape Nov 2025 results
- Paper trade backtest
- If fails → Stop or retrain

### Week 3-4: Robustness Testing
- Walk-forward validation
- Parameter sensitivity
- Stress testing

### Week 5-6: Risk Assessment
- Monte Carlo simulations
- Determine bet sizing
- Set stop-loss rules

### Week 7-8: Live Paper Trading
- Make predictions daily
- Track performance
- No real money yet

### Week 9+: Small Live Betting (if all checks pass)
- Start with 1/8 Kelly sizing
- Min bet sizes
- Track everything
- Scale up only after sustained success

## Bottom Line

**DO NOT BET REAL MONEY UNTIL**:
1. You validate on Nov 2025 actual results
2. Walk-forward validation shows consistency
3. Paper trading shows sustained profitability
4. Monte Carlo shows acceptable risk

**Your concern about overfitting is valid and responsible. The fact that you're questioning it means you're thinking correctly about this.**
