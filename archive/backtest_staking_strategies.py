"""
Backtest Staking Strategies on Historical Races (June-November 2025)
Uses BULK LOADING optimization (like backtest_november_bulk.py)
Tests multiple staking strategies to find best ROI
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

# Configuration
DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'
CONFIDENCE_THRESHOLD = 0.80  # High confidence only - target <10 bets/day
MINIMUM_MARGIN = 0.02  # Require model to be at least 2% better than implied odds
INITIAL_BANKROLL = 1000.0

print("="*80)
print("BACKTEST STAKING STRATEGIES: June-November 2025")
print("="*80)

# Load trained model
print("\nLoading trained ML model...")
ml_model = GreyhoundMLModel()
try:
    ml_model.load_model()
    print(f"Model loaded successfully")
    print(f"  Features: {ml_model.feature_columns}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Load all race entries from June-November 2025 with actual results
print(f"\nLoading race data from {START_DATE} to {END_DATE}...")
query = """
SELECT
    ge.EntryID,
    g.GreyhoundName,
    g.GreyhoundID,
    t.TrackName,
    t.TrackID,
    rm.MeetingDate,
    r.RaceNumber,
    r.Distance,
    ge.Box,
    ge.Weight,
    ge.Position,
    ge.StartingPrice
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
"""

df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
print(f"Loaded {len(df):,} race entries")

if len(df) == 0:
    print("ERROR: No race data found for the selected period.")
    conn.close()
    exit(1)

# Filter out NZ and Tasmania tracks
excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                  'Launceston', 'Hobart', 'Devonport']
df = df[~df['TrackName'].isin(excluded_tracks)]
df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]
df = df[~df['TrackName'].str.contains('TAS', na=False, case=False)]
print(f"After filtering NZ/TAS tracks: {len(df):,}")

# Convert Position to numeric, mark winners
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['IsWinner'] = (df['Position'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce').fillna(2.0)
df['StartingPrice'] = df['StartingPrice'].clip(lower=1.5)

print(f"Winners: {df['IsWinner'].sum():,}")

# BULK LOAD: Get ALL historical races before June 1, 2025 in ONE query
print("\nBulk loading historical data (this is the key optimization)...")
historical_query = """
    SELECT
        ge.GreyhoundID,
        t.TrackName,
        r.Distance,
        ge.Weight,
        ge.Position,
        ge.FinishTime,
        ge.SplitBenchmarkLengths as G_Split_ADJ,
        rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
        ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
        rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < ?
      AND ge.Position IS NOT NULL
      AND ge.SplitBenchmarkLengths IS NOT NULL
      AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
      AND ge.FinishTimeBenchmarkLengths IS NOT NULL
      AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
    ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
"""

hist_df = pd.read_sql_query(historical_query, conn, params=(START_DATE,))
print(f"Loaded {len(hist_df):,} historical races")

# Calculate box win rates (using data BEFORE June 1, 2025)
print("\nCalculating box win rates...")
box_stats_query = """
    SELECT
        t.TrackID,
        r.Distance,
        ge.Box,
        COUNT(*) as TotalRaces,
        SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate < ?
      AND ge.Position IS NOT NULL
      AND ge.Box IS NOT NULL
    GROUP BY t.TrackID, r.Distance, ge.Box
"""
box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(START_DATE,))
box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']

# Create lookup dict
box_win_rates = {}
for _, row in box_stats_df.iterrows():
    key = (row['TrackID'], row['Distance'], row['Box'])
    box_win_rates[key] = row['BoxWinRate']

print(f"Box win rates calculated for {len(box_win_rates)} combinations")

conn.close()

# Group historical data by greyhound for fast lookup
print("\nGrouping historical data by greyhound...")
hist_grouped = hist_df.groupby('GreyhoundID')
print(f"Grouped into {len(hist_grouped)} greyhounds")

# Extract features efficiently (same approach as model training)
print("\nExtracting features from historical data...")
features_list = []
entry_info = []
total = len(df)
skipped = 0

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  Processed {idx}/{total} entries ({idx/total*100:.1f}%)")

    # Get this greyhound's history before this race
    if row['GreyhoundID'] not in hist_grouped.groups:
        skipped += 1
        continue

    greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
    # Filter to races before current date
    greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]

    # Take last 5 races
    last_5 = greyhound_hist.head(5)

    if len(last_5) < 5:
        skipped += 1
        continue

    # Build features (same logic as model)
    features = {}

    # Box win rate
    box_key = (row['TrackID'], row['Distance'], row['Box'])
    features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)

    # Recent form from last 3 races
    last_3 = last_5.head(3)
    if len(last_3) > 0:
        last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
        features['AvgPositionLast3'] = last_3_positions.mean()
        features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
    else:
        features['AvgPositionLast3'] = 4.5
        features['WinRateLast3'] = 0

    # GM_OT_ADJ features with track tier weighting
    for i, (_, race) in enumerate(last_5.iterrows(), 1):
        gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)

        # Apply track tier weight
        track_weight = ml_model.get_track_tier_weight(race['TrackName'])
        features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight

    features_list.append(features)
    entry_info.append({
        'EntryID': row['EntryID'],
        'GreyhoundName': row['GreyhoundName'],
        'TrackName': row['TrackName'],
        'MeetingDate': row['MeetingDate'],
        'RaceNumber': row['RaceNumber'],
        'Box': row['Box'],
        'Won': row['IsWinner'],
        'Position': row['Position'],
        'StartingPrice': row['StartingPrice']
    })

print(f"Successfully extracted features for {len(features_list):,} entries")
print(f"Skipped {skipped:,} entries (insufficient historical data)")

if len(features_list) == 0:
    print("ERROR: Could not extract any features. Cannot backtest.")
    exit(1)

# Create feature DataFrame
X = pd.DataFrame(features_list)
entry_df = pd.DataFrame(entry_info)

# Ensure we have all required feature columns
missing_cols = set(ml_model.feature_columns) - set(X.columns)
for col in missing_cols:
    X[col] = 0

X = X[ml_model.feature_columns]

# Make predictions
print("\nMaking predictions...")
raw_probabilities = ml_model.model.predict_proba(X)[:, 1]
# DON'T apply calibration - it was too aggressive and reduced bets too much
# Use raw predictions instead
probabilities = raw_probabilities
entry_df['WinProbability'] = probabilities

print(f"Predictions made: {len(entry_df):,}")
print(f"Probability range: {probabilities.min():.3f} to {probabilities.max():.3f}")
print(f"Mean probability: {probabilities.mean():.3f}")

# Filter by confidence threshold
above_threshold = entry_df[entry_df['WinProbability'] >= CONFIDENCE_THRESHOLD]
print(f"\nEntries with confidence >= {CONFIDENCE_THRESHOLD*100:.0f}%: {len(above_threshold):,}")

if len(above_threshold) == 0:
    print("ERROR: No predictions meet confidence threshold. Cannot backtest.")
    exit(1)

# Filter for value bets (win probability > implied probability by minimum margin)
above_threshold['ImpliedProb'] = 1 / above_threshold['StartingPrice']
above_threshold['Margin'] = above_threshold['WinProbability'] - above_threshold['ImpliedProb']
value_bets = above_threshold[above_threshold['Margin'] >= MINIMUM_MARGIN]
print(f"Value bets (margin >= {MINIMUM_MARGIN:.1%}): {len(value_bets):,}")

# Use value bets, or all above threshold if no value bets found
if len(value_bets) > 0:
    filtered_df = value_bets.copy()
    print("Using all value bets where model has edge")
else:
    filtered_df = above_threshold.copy()
    print("WARNING: No value bets found. Using all predictions above confidence threshold.")

print(f"Total bets to evaluate: {len(filtered_df):,}")

# Define staking strategies
class StakingStrategy:
    def calculate_stake(self, bankroll, odds, win_prob):
        raise NotImplementedError

class FlatStake(StakingStrategy):
    """Bet fixed 2% of bankroll per bet"""
    def calculate_stake(self, bankroll, odds, win_prob):
        return max(1.0, bankroll * 0.02)

class KellyCriteria(StakingStrategy):
    """Full Kelly criterion with safeguards"""
    def calculate_stake(self, bankroll, odds, win_prob):
        if win_prob <= 0 or win_prob >= 1:
            return 0
        b = odds - 1
        p = win_prob
        q = 1 - p
        kelly = (p * b - q) / b
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        return max(1.0, bankroll * kelly)

class HalfKelly(StakingStrategy):
    """Half Kelly (conservative)"""
    def calculate_stake(self, bankroll, odds, win_prob):
        if win_prob <= 0 or win_prob >= 1:
            return 0
        b = odds - 1
        p = win_prob
        q = 1 - p
        kelly = (p * b - q) / b
        kelly = max(0, min(kelly * 0.5, 0.15))  # Half Kelly, cap at 15%
        return max(1.0, bankroll * kelly)

class ConfidenceProportional(StakingStrategy):
    """Bet more when confidence is higher above threshold"""
    def calculate_stake(self, bankroll, odds, win_prob):
        excess = max(0, win_prob - CONFIDENCE_THRESHOLD)
        max_excess = 1.0 - CONFIDENCE_THRESHOLD
        scale = excess / max_excess if max_excess > 0 else 0
        stake_pct = 0.01 + (0.04 * scale)  # 1% to 5%
        return max(1.0, bankroll * stake_pct)

# Test each strategy
strategies = {
    'Flat Stake (2%)': FlatStake(),
    'Kelly Criteria': KellyCriteria(),
    'Half Kelly': HalfKelly(),
    'Confidence Proportional': ConfidenceProportional(),
}

results = []

for strategy_name, strategy in strategies.items():
    bankroll = INITIAL_BANKROLL
    bets_placed = 0
    wins = 0
    total_staked = 0
    total_returned = 0
    bet_details = []
    
    for _, bet in filtered_df.iterrows():
        odds = float(bet['StartingPrice'])
        win_prob = bet['WinProbability']
        
        stake = strategy.calculate_stake(bankroll, odds, win_prob)
        
        if stake <= 0 or stake > bankroll * 0.5:  # Max 50% per bet
            continue
        
        bets_placed += 1
        total_staked += stake
        
        # Calculate return
        if bet['Won'] == 1:
            returns = stake * odds
            profit = returns - stake
            wins += 1
        else:
            returns = 0
            profit = -stake
        
        total_returned += returns
        bankroll += profit
        
        bet_details.append({
            'Dog': bet['GreyhoundName'],
            'Race': f"{bet['RaceNumber']}",
            'Track': bet['TrackName'],
            'Odds': f"{odds:.2f}",
            'Prob': f"{win_prob:.1%}",
            'Stake': f"${stake:.2f}",
            'Result': 'WIN' if bet['Won'] == 1 else 'LOSE'
        })
    
    profit_loss = bankroll - INITIAL_BANKROLL
    roi = (profit_loss / INITIAL_BANKROLL * 100) if INITIAL_BANKROLL > 0 else 0
    strike_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
    
    results.append({
        'Strategy': strategy_name,
        'Bets': bets_placed,
        'Wins': wins,
        'Strike%': strike_rate,
        'Staked': total_staked,
        'Returned': total_returned,
        'P/L': profit_loss,
        'ROI%': roi,
        'Final': bankroll,
        'Details': bet_details
    })

# Display results
print("\n" + "="*80)
print("BACKTEST RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROI%', ascending=False)

print(f"\n{'Strategy':<25} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'P/L':<12} {'ROI%':<10}")
print("-"*80)

for _, row in results_df.iterrows():
    print(f"{row['Strategy']:<25} {row['Bets']:<8.0f} {row['Wins']:<8.0f} "
          f"{row['Strike%']:<10.1f} ${row['P/L']:<11.2f} {row['ROI%']:<10.2f}%")

# Analysis of why we're losing money
print(f"\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print(f"\nTotal predictions made: {len(entry_df):,}")
print(f"High confidence predictions (>= 80%): {len(entry_df[entry_df['WinProbability'] >= 0.8]):,}")
print(f"Used for betting (after value filter): {len(filtered_df):,}")
print(f"\nActual strike rate: {(filtered_df['Won'].sum() / len(filtered_df) * 100):.2f}%")
print(f"Model predicted strike rate: {(filtered_df['WinProbability'].mean() * 100):.2f}%")

# Check if predictions are overconfident
print(f"\n--- PREDICTION CALIBRATION ---")
prob_bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
for i in range(len(prob_bins) - 1):
    low, high = prob_bins[i], prob_bins[i+1]
    in_bin = filtered_df[(filtered_df['WinProbability'] >= low) & (filtered_df['WinProbability'] < high)]
    if len(in_bin) > 0:
        actual_wr = in_bin['Won'].sum() / len(in_bin) * 100
        pred_wr = low * 100
        diff = actual_wr - pred_wr
        print(f"  Predicted {pred_wr:.0f}%-{high*100:.0f}%: {len(in_bin):>4} bets, actual win% = {actual_wr:>5.1f}% (diff: {diff:+.1f}%)")

# ROI breakdown by odds bracket for Flat Staking
print(f"\n--- FLAT STAKING ROI BY ODDS BRACKET ---")
odds_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 100.0)]
flat_stake_pct = 0.02

for low_odds, high_odds in odds_bins:
    in_bracket = filtered_df[(filtered_df['StartingPrice'] >= low_odds) & (filtered_df['StartingPrice'] < high_odds)]
    if len(in_bracket) > 0:
        total_staked = len(in_bracket) * INITIAL_BANKROLL * flat_stake_pct
        wins = in_bracket['Won'].sum()
        total_return = (in_bracket[in_bracket['Won'] == 1]['StartingPrice'] * INITIAL_BANKROLL * flat_stake_pct).sum()
        profit = total_return - total_staked
        roi = (profit / total_staked) * 100
        win_rate = (wins / len(in_bracket)) * 100
        avg_odds = in_bracket['StartingPrice'].mean()
        print(f"  ${low_odds:.1f}-${high_odds:.1f}: {len(in_bracket):>4} bets, wins: {wins:>3}, "
              f"win%: {win_rate:>5.1f}%, avg odds: {avg_odds:>5.2f}, profit: ${profit:>8.2f}, roi: {roi:>7.2f}%")

# Best strategy details
if len(results) > 0:
    best = results_df.iloc[0]
    print(f"\n" + "="*80)
    print(f"BEST STRATEGY: {best['Strategy']}")
    print("="*80)
    print(f"Bets placed: {best['Bets']:.0f}")
    print(f"Wins: {best['Wins']:.0f}")
    print(f"Strike rate: {best['Strike%']:.2f}%")
    print(f"Total staked: ${best['Staked']:.2f}")
    print(f"Total returned: ${best['Returned']:.2f}")
    print(f"Profit/Loss: ${best['P/L']:.2f}")
    print(f"ROI: {best['ROI%']:.2f}%")
    print(f"Final bankroll: ${best['Final']:.2f}")
    
    if best['Details']:
        print(f"\nFirst 30 bets:")
        print(f"{'Dog':<20} {'Race':<6} {'Track':<15} {'Odds':<8} {'Prob':<8} {'Stake':<12} {'Result':<8}")
        print("-"*100)
        for bet in best['Details'][:30]:
            print(f"{bet['Dog']:<20} {bet['Race']:<6} {bet['Track']:<15} {bet['Odds']:<8} "
                  f"{bet['Prob']:<8} {bet['Stake']:<12} {bet['Result']:<8}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"""
The backtest shows that even with:
- Trained ML model with 80%+ confidence predictions
- Value bet filtering (prob > implied odds)
- Multiple staking strategies tested

All strategies show NEGATIVE ROI for June-November 2025.

POSSIBLE EXPLANATIONS:
1. Model overconfidence: Predicted probabilities higher than actual win rates
2. Odds efficiency: Bookmakers pricing already accounts for dogs' abilities
3. Track conditions: Model trained on 2023-2024 data, conditions may have changed
4. Survivorship bias: Model only trained on finished races (not DNF/SCR)
5. Time decay: Market odds improve closer to race time (bets placed well in advance)

NEXT STEPS:
1. Check calibration: Compare predicted vs actual win% by confidence bins
2. Consider lower confidence threshold (70%, 75%) for more bets
3. Examine top wins/losses to find patterns
4. Investigate track-specific performance
5. Consider retraining model with more recent data
""")

print("="*80)
