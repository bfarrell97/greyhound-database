# Why The Model Is Unprofitable (And How To Fix It)

## The Problem Analysis

### What We Discovered
1. **Unfixed model (80% confidence threshold):**
   - 552 bets placed
   - 35.5% strike rate
   - **-69.7% ROI** (lost $696.75 from $1000 bankroll)
   - Model predicted 82.98% strike rate but only achieved 35.5% ← MASSIVE miscalibration

2. **Short odds filtering ($1.50-$2.00 only):**
   - 2,311 bets placed  
   - 53.5% strike rate (much better!)
   - **-99.9% ROI** (lost $998.97 from $1000 bankroll)
   - The winning bets still don't compensate for losses

### Why It Fails
At $1.69 average odds with 55.4% win rate:
- 1000 bets × $1 stake = $1000 staked
- 554 wins × $1.69 = $936 returned
- **Loss: $64 per 1000 bets = 6.4% negative expected value**

Even with **55% accuracy on favorites**, the bookmakers have the odds set correctly so that you still lose money. This means:
- **The model doesn't have an edge at ANY odds**
- **Bookmakers know something the model doesn't**

---

## Root Cause: Missing Features

The model currently uses:
- `BoxWinRate` - Statistical track/distance/box performance
- `AvgPositionLast3` - Average finishing position in last 3 races
- `WinRateLast3` - Win rate in last 3 races
- `GM_OT_ADJ_1-5` - Benchmark times weighted by track tier

**What's missing:**
1. **Market efficiency signal** - Does the bookmaker disagree with the model? That matters!
2. **Track bias patterns** - Some tracks heavily favor certain racing styles (inside runners, front runners, chasers)
3. **Form recency** - Last race matters more than 5 races ago, but current model treats equally
4. **Opposition quality** - Beating weaklings ≠ beating strong fields
5. **Weather/track condition effects** - Wet vs dry, soft vs firm track handling
6. **Specific dog characteristics** - Some dogs peak at specific distances/tracks
7. **Draw patterns** - Certain box draws consistently outperform on specific tracks
8. **Sectional times** - 300m time vs 600m time reveal different dog types

---

## The Solution: Feature Engineering

### Phase 1: Add Market Confidence (Quick Win)
This won't fix the profitability but will help the model learn what the market knows:

```python
# In feature extraction:
entry_df['BookmakerProb'] = 1 / entry_df['StartingPrice']
entry_df['ModelVsMarket'] = entry_df['WinProbability'] - entry_df['BookmakerProb']

# Add to training features
X['MarketConfidence'] = 1 / df['StartingPrice']
X['ModelMarketDiff'] = X['PredictedProb'] - X['MarketConfidence']
```

**Why:** When model predicts 60% but market prices 40%, that disagreement is valuable information. The model should learn to either trust its edge (and profit) or defer to market (and stay neutral).

### Phase 2: Add Track Bias Features (Medium Effort)
Track-specific statistics that persist:

```python
# Calculate for each track:
# - Box draw win rates: Does box 1 win 25% vs 12.5% baseline?
# - Running style advantage: Do front-runners win more here?
# - Distance trends: Is the track a mile-specialist track?
# - Weather patterns: Does this track play differently on wet days?

track_biases = {
    'Angle Park': {'front_runner_bonus': 0.05, 'box_1_advantage': 1.3},
    'Sale': {'chaser_bonus': 0.08, 'box_4_advantage': 1.4},
    # ... etc for each track
}

# In feature extraction:
X['TrackFrontRunnerBonus'] = dog_style * track_biases[track].get('front_runner_bonus', 0)
```

### Phase 3: Recency Weighting (Medium Effort)
Weight recent form higher:

```python
# When calculating AvgPositionLast3 and WinRateLast3
# Weight last race 3x, second-last 2x, third-last 1x (instead of equal)
weights = [3, 2, 1]
recent_form = weighted_avg(last_3_positions, weights)
```

### Phase 4: Opposition Quality Score (Hard)
Account for strength of competition:

```python
# For each race the dog ran in:
opposition_strength = avg_rating_of_other_dogs_in_race
dog_relative_strength = dog_rating / opposition_strength

# A win at 55% relative strength > win at 90% relative strength
```

---

## Implementation Priority

**Do First (Quick Wins):**
1. ✅ Add `BookmakerProb` and `ModelVsMarket` features
2. ✅ Implement recency weighting in form calculation
3. ✅ Add per-track box draw statistics

**Do After (Better but Slower):**
4. Track bias features (front-runner bonus, chaser advantage)
5. Opposition quality scoring
6. Weather/condition handling

**Do If Time (Advanced):**
7. Sectional times analysis
8. Dog-specific distance/track affinity
9. Complex interaction terms

---

## Expected Impact

**Conservative estimate with Phase 1-3 improvements:**
- Strike rate on favorites: **60-65%** (up from 53%)
- With better feature engineering reducing model error
- Combined with calibration: Could reach **+5 to +15% ROI**

**Aggressive estimate with all phases:**
- Strike rate: **65-70%**
- Better odds: Willing to bet at +2.5 to +4.0 on found edges
- Expected ROI: **+20-40%** annually on carefully selected bets

---

## Next Steps

1. Add BookmakerProb + ModelVsMarket features to greyhound_ml_model.py
2. Retrain model on 2023-2024 data with new features
3. Use Platt scaling to properly calibrate probabilities on validation set
4. Backtest retrained model
5. If still unprofitable, add track bias features and retrain again

The key insight: **Odds filtering alone won't work. You need to improve the model's understanding of what differentiates dogs at each odds level.**
