import pandas as pd
import numpy as np

def run_monte_carlo(profits, simulations=10000):
    max_drawdowns = []
    
    for _ in range(simulations):
        shuffled_profits = np.random.permutation(profits)
        cum_profit = np.cumsum(shuffled_profits)
        equity_curve = np.insert(cum_profit, 0, 0)
        peak_equity = np.maximum.accumulate(equity_curve)
        drawdowns = peak_equity - equity_curve
        max_drawdowns.append(np.max(drawdowns))
        
    return np.mean(max_drawdowns), np.percentile(max_drawdowns, 95)

def compare_staking(csv_file):
    print(f"Loading bets from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    # Inverse Staking (Existing)
    inverse_profit = df['Profit'].sum()
    inverse_stake = df['Stake'].sum()
    inverse_roi = (inverse_profit / inverse_stake) * 100
    
    print("\nRunning Monte Carlo for Inverse Staking...")
    inv_avg_dd, inv_95_dd = run_monte_carlo(df['Profit'].values)
    
    # Flat Staking (Calculated)
    # Profit = (Odds - 1) * 1 if Win, else -1
    df['FlatProfit'] = df.apply(lambda row: (row['CurrentOdds'] - 1) if row['IsWinner'] else -1, axis=1)
    
    flat_profit = df['FlatProfit'].sum()
    flat_stake = len(df) # 1 unit per bet
    flat_roi = (flat_profit / flat_stake) * 100
    
    print("Running Monte Carlo for Flat Staking...")
    flat_avg_dd, flat_95_dd = run_monte_carlo(df['FlatProfit'].values)
    
    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'INVERSE STAKING':<20} | {'FLAT STAKING':<20} | {'DIFF':<10}")
    print("="*80)
    print(f"{'Total Profit':<20} | {inverse_profit:>18.2f}u | {flat_profit:>18.2f}u | {flat_profit - inverse_profit:>+9.2f}u")
    print(f"{'ROI':<20} | {inverse_roi:>18.2f}% | {flat_roi:>18.2f}% | {flat_roi - inverse_roi:>+9.2f}%")
    print("-" * 80)
    print(f"{'Avg Drawdown':<20} | {inv_avg_dd:>18.2f}u | {flat_avg_dd:>18.2f}u | {flat_avg_dd - inv_avg_dd:>+9.2f}u")
    print(f"{'95% Max Drawdown':<20} | {inv_95_dd:>18.2f}u | {flat_95_dd:>18.2f}u | {flat_95_dd - inv_95_dd:>+9.2f}u")
    print("="*80)

if __name__ == "__main__":
    compare_staking('backtest_production_leader.csv')
