import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

def stress_test(csv_file, simulations=10000):
    print(f"Loading bets from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    if len(df) == 0:
        print("No bets to test.")
        return

    print(f"Loaded {len(df)} bets.")
    
    # Calculate key metrics
    wins = df['IsWinner'].sum()
    strike_rate = (wins / len(df)) * 100
    avg_odds = df['CurrentOdds'].mean()
    
    # Actual Result
    actual_profit = df['Profit'].sum()
    actual_roi = (actual_profit / df['Stake'].sum()) * 100
    
    print("\n" + "="*50)
    print("BASELINE PERFORMANCE")
    print("="*50)
    print(f"Bets: {len(df)}")
    print(f"Strike Rate: {strike_rate:.2f}%")
    print(f"Avg Odds: ${avg_odds:.2f}")
    print(f"Total Profit: {actual_profit:.2f} units")
    print(f"ROI: {actual_roi:.2f}%")
    
    # Monte Carlo Logic
    # We resample the Sequence of Wins/Losses (to test luck/variance)
    # BUT we should respect the odds.
    # Method 1: Shuffle the actual outcomes (re-ordering). 
    #   - Tests drawdown and path dependency.
    #   - Does not test "what if I was slightly unluckier with the same odds"?
    # Method 2: Resample based on Implied Probability (or True Probability estimate).
    #   - Better for checking if the edge is real.
    
    # Let's do Method 1 (Permutation) for Drawdown Analysis
    print("\n" + "="*50)
    print(f"MONTE CARLO SIMULATION ({simulations} runs)")
    print("="*50)
    print("Permuting trade sequence to estimate risk...")
    
    profits = df['Profit'].values
    stakes = df['Stake'].values
    
    mc_rois = []
    max_drawdowns = []
    
    for _ in range(simulations):
        # Shuffle profits
        shuffled_profits = np.random.permutation(profits)
        cum_profit = np.cumsum(shuffled_profits)
        
        # ROI is same for all permutations (sum is invariant), 
        # but Drawdown depends on order.
        
        # Calculate Max Drawdown
        peak = np.maximum.accumulate(np.insert(cum_profit, 0, 0))
        # Drawdown = Peak - Current (positive value)
        # We want the max of that
        # Insert 0 at start to handle immediate drawdown
        drawdown = peak[:-1] - cum_profit
        # Wait, peak is array of len N+1?
        # cum_profit is len N.
        # peak corresponds to 0..N.
        # drawdown should be peak[i] - cum_profit[i] (where current equity = cum_profit[i])
        
        # Correct calculation:
        equity_curve = np.insert(cum_profit, 0, 0)
        peak_equity = np.maximum.accumulate(equity_curve)
        drawdowns = peak_equity - equity_curve
        max_dd = np.max(drawdowns)
        max_drawdowns.append(max_dd)

    avg_dd = np.mean(max_drawdowns)
    p95_dd = np.percentile(max_drawdowns, 95)
    p99_dd = np.percentile(max_drawdowns, 99)
    
    print(f"Average Max Drawdown: {avg_dd:.2f} units")
    print(f"95% Worst Case Drawdown: {p95_dd:.2f} units")
    print(f"99% Worst Case Drawdown: {p99_dd:.2f} units")
    
    # Method 2: Bootstrap Resampling (with replacement)
    # This tests the "Robustness" of the Edge.
    # Can we lose money over N bets just by bad sampling?
    
    print("\nBootstrapping (Resampling with replacement)...")
    bootstrap_profits = []
    losing_runs = 0
    
    for _ in range(simulations):
        # Sample N bets with replacement
        indices = np.random.choice(len(df), len(df), replace=True)
        sample_profits = profits[indices]
        sample_stakes = stakes[indices]
        
        total_profit = np.sum(sample_profits)
        total_stake = np.sum(sample_stakes)
        
        roi = str_roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        bootstrap_profits.append(total_profit)
        
        if total_profit < 0:
            losing_runs += 1

    prob_loss = (losing_runs / simulations) * 100
    avg_boot_profit = np.mean(bootstrap_profits)
    
    print(f"Probability of Loss over {len(df)} bets: {prob_loss:.2f}%")
    print(f"Average Bootstrap Profit: {avg_boot_profit:.2f} units")
    
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    if prob_loss < 5:
        print("PASS: Strategy is robust (<5% chance of loss)")
    elif prob_loss < 20:
        print("WARNING: Moderate risk of loss (5-20%)")
    else:
        print("FAIL: High risk of loss (>20%)")
        
    if p95_dd > 50:
        print(f"WARNING: High drawdown potential ({p95_dd:.2f}u)")
    else:
        print(f"PASS: Drawdown acceptable ({p95_dd:.2f}u)")

if __name__ == "__main__":
    stress_test('backtest_production_leader.csv')
