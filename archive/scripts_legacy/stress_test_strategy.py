import pandas as pd
import numpy as np
import random

def run_stress_test():
    print("Loading trades...")
    try:
        df = pd.read_csv("all_trades.csv")
    except FileNotFoundError:
        print("all_trades.csv not found. Run verifiction script first.")
        return

    print(f"Loaded {len(df)} trades.")
    
    # Calculate PnL per trade
    # Win (Status=0) -> +0.95 * Stake
    # Loss (Status=1) -> -(Odds-1) * Stake
    STAKE = 100
    COMM = 0.05
    
    # Calculate Liability (needed for ROI base)
    df['Liability'] = (df['Odds'] - 1) * STAKE

    # Vectorized PnL
    df['PnL'] = np.where(
        df['IsWin'] == 0,
        STAKE * (1 - COMM),
        -1 * (df['Odds'] - 1) * STAKE
    )
    
    real_roi = df['PnL'].sum() / df['Liability'].sum() * 100
    print(f"Realized ROI: {real_roi:.2f}%")
    
    trades = df['PnL'].values
    
    print("\n--- 1. Monte Carlo Simulation (Risk of Ruin) ---")
    # Shuffle trades 10,000 times. Check if Bankroll < 0.
    # Bankroll = 50 units = 50 * 100 = $5000.
    START_BANK = 50 * STAKE 
    N_SIMS = 10000
    ruined_count = 0
    worst_drawdowns = []
    
    for _ in range(N_SIMS):
        shuffled = np.random.permutation(trades)
        equity_curve = np.cumsum(shuffled)
        
        # Check Ruin (Equity < -START_BANK) -> effectively 0 funds
        if np.min(equity_curve) < -START_BANK:
            ruined_count += 1
            
        # Drawdown
        # Peak-to-valley
        # This is absolute $ drawdown from high water mark.
        # But commonly we check Max Drawdown as % or units.
        # For simplicity: Max Units Drawdown from Peak.
        
        running_max = np.maximum.accumulate(np.insert(equity_curve, 0, 0))
        # equity_curve with 0 start
        curve_with_0 = np.insert(equity_curve, 0, 0)
        drawdowns = running_max - curve_with_0
        worst_drawdowns.append(np.max(drawdowns) / STAKE) # In Units
        
    risk_of_ruin = ruined_count / N_SIMS * 100
    p99_dd = np.percentile(worst_drawdowns, 99)
    
    print(f"Index: Risk of Ruin (50u Bank): {risk_of_ruin:.2f}%")
    print(f"Worst-case Drawdown (99th %): {p99_dd:.1f} units")
    
    print("\n--- 2. Bootstrap Resampling (ROI Confidence) ---")
    # Resample with replacement 10,000 times
    means = []
    
    # Total Liability is needed for ROI, not just pure PnL avg.
    # We must bootstrap pairs (PnL, Liability).
    
    liabilities = df['Liability'].values
    indices = np.arange(len(trades))
    
    rois = []
    for _ in range(N_SIMS):
        # Resample indices
        # Optimization: use numpy choice
        idx = np.random.choice(indices, size=len(indices), replace=True)
        
        boot_pnl = trades[idx].sum()
        boot_risk = liabilities[idx].sum()
        
        rois.append(boot_pnl / boot_risk * 100)
        
    ci_lower = np.percentile(rois, 2.5)
    ci_upper = np.percentile(rois, 97.5)
    
    print(f"95% Confidence Interval: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    print(f"Prob(ROI > 10%): {(np.array(rois) > 10).mean() * 100:.1f}%")
    
if __name__ == "__main__":
    run_stress_test()
