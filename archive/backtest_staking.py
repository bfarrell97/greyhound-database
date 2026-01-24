"""
Backtest Staking Strategies on Historical Races (June-November 2025)
Uses trained ML model's predict_race_winners method for efficiency
Tests multiple staking strategies to find best ROI
"""

import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel
import os

# Configuration
DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-06-01'
END_DATE = '2025-11-30'
CONFIDENCE_THRESHOLD = 0.80
INITIAL_BANKROLL = 1000.0

print("="*80)
print("BACKTEST STAKING STRATEGIES: June-November 2025")
print("="*80)

# Load trained model
if not os.path.exists('greyhound_model.pkl'):
    print("ERROR: No trained model found. Train the model first in the GUI.")
    exit(1)

print("\nLoading trained ML model...")
ml_model = GreyhoundMLModel()
try:
    ml_model.load_model()
    print(f"Model loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

# Get unique race dates
conn = sqlite3.connect(DB_PATH)
dates_query = """
SELECT DISTINCT rm.MeetingDate
FROM RaceMeetings rm
WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
ORDER BY rm.MeetingDate
"""
dates_df = pd.read_sql_query(dates_query, conn, params=(START_DATE, END_DATE))
conn.close()

if len(dates_df) == 0:
    print(f"ERROR: No race meetings found for {START_DATE} to {END_DATE}")
    exit(1)

print(f"Found {len(dates_df)} race dates to process")

# Collect all predictions using model's optimized method
print("\nGenerating predictions for all races...")
all_predictions = []

for idx, row in dates_df.iterrows():
    meeting_date = row['MeetingDate']
    if idx % 20 == 0:
        print(f"  Processing {idx+1}/{len(dates_df)}: {meeting_date}")
    
    try:
        # Use model's optimized prediction method with 0 threshold to get all entries
        predictions = ml_model.predict_race_winners(meeting_date, confidence_threshold=0.0)
        if len(predictions) > 0:
            all_predictions.append(predictions)
    except Exception as e:
        # Silently skip dates with errors
        pass

if not all_predictions:
    print("ERROR: No predictions generated.")
    exit(1)

# Combine all predictions
df = pd.concat(all_predictions, ignore_index=True)
print(f"Generated predictions for {len(df)} race entries")

# Load actual race results
print("\nLoading actual race results...")
conn = sqlite3.connect(DB_PATH)
results_query = """
SELECT 
    ge.EntryID,
    ge.Position
FROM GreyhoundEntries ge
"""
results_df = pd.read_sql_query(results_query, conn)
conn.close()

# Merge results
df = df.merge(results_df[['EntryID', 'Position']], on='EntryID', how='left')

# Prepare data
df['PositionNum'] = pd.to_numeric(df['Position'], errors='coerce')
df['Won'] = (df['PositionNum'] == 1).astype(int)
df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce').fillna(2.0)
df['StartingPrice'] = df['StartingPrice'].clip(lower=1.5)

print(f"Merged with {df['Won'].sum():,} winners out of {len(df):,} entries")

# Filter by confidence threshold and value bets
df['ImpliedProb'] = 1 / df['StartingPrice']
df['ModelValue'] = df['WinProbability'] * df['StartingPrice']

above_threshold = len(df[df['WinProbability'] >= CONFIDENCE_THRESHOLD])
value_bets = len(df[(df['WinProbability'] >= CONFIDENCE_THRESHOLD) & 
                    (df['WinProbability'] > df['ImpliedProb'])])

print(f"\nFiltering results:")
print(f"  Entries with confidence >= {CONFIDENCE_THRESHOLD*100:.0f}%: {above_threshold:,}")
print(f"  Value bets (prob > implied): {value_bets:,}")

# Use bets that meet both criteria
filtered_df = df[(df['WinProbability'] >= CONFIDENCE_THRESHOLD) & 
                 (df['WinProbability'] > df['ImpliedProb'])].copy()

if len(filtered_df) == 0:
    print("\nWARNING: No bets meet confidence AND value criteria.")
    print("Using confidence threshold only...")
    filtered_df = df[df['WinProbability'] >= CONFIDENCE_THRESHOLD].copy()

if len(filtered_df) == 0:
    print("ERROR: No bets qualify. Exiting.")
    exit(1)

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
            'Track': bet['CurrentTrack'],
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
print("BACKTEST RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROI%', ascending=False)

print(f"\n{'Strategy':<25} {'Bets':<8} {'Wins':<8} {'Strike%':<10} {'P/L':<12} {'ROI%':<10}")
print("-"*85)

for _, row in results_df.iterrows():
    print(f"{row['Strategy']:<25} {row['Bets']:<8.0f} {row['Wins']:<8.0f} "
          f"{row['Strike%']:<10.1f} ${row['P/L']:<11.2f} {row['ROI%']:<10.2f}%")

# Show best strategy details
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
        print(f"\nFirst 20 bets:")
        print(f"{'Dog':<20} {'Race':<6} {'Track':<15} {'Odds':<8} {'Prob':<8} {'Stake':<12} {'Result':<8}")
        print("-"*100)
        for bet in best['Details'][:20]:
            print(f"{bet['Dog']:<20} {bet['Race']:<6} {bet['Track']:<15} {bet['Odds']:<8} "
                  f"{bet['Prob']:<8} {bet['Stake']:<12} {bet['Result']:<8}")

print("\n" + "="*80)
