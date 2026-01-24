import pandas as pd
import numpy as np

def get_stake_audit(odds):
    # Re-implementing staking logic completely independent of original script
    if odds < 3: return 0.5
    elif odds < 5: return 0.75
    elif odds < 10: return 1.0
    elif odds < 20: return 1.5
    else: return 2.0

def audit_file(filename):
    print(f"\nAUDITING: {filename}")
    print("="*60)
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("  [ERROR] File not found.")
        return

    print(f"  Rows loaded: {len(df)}")
    
    errors = 0
    epsilon = 0.0001 # Float tolerance
    
    # 1. Audit Staking Logic
    calculated_stakes = df['CurrentOdds'].apply(get_stake_audit)
    stake_diffs = np.abs(df['Stake'] - calculated_stakes)
    stake_errors = stake_diffs[stake_diffs > epsilon]
    if len(stake_errors) > 0:
        print(f"  [FAIL] Staking Logic Mismatch in {len(stake_errors)} rows.")
        errors += 1
    else:
        print("  [PASS] Staking Logic matches.")
        
    # 2. Audit Return Logic
    # Expected Return = Stake * Odds (if Win) else 0
    calc_return = np.where(df['IsWinner'] == 1, df['Stake'] * df['CurrentOdds'], 0)
    return_diffs = np.abs(df['Return'] - calc_return)
    return_errors = return_diffs[return_diffs > epsilon]
    if len(return_errors) > 0:
        print(f"  [FAIL] Return Logic Mismatch in {len(return_errors)} rows.")
        errors += 1
    else:
        print("  [PASS] Return Logic matches.")
        
    # 3. Audit Profit Logic
    # Expected Profit = Return - Stake
    calc_profit = df['Return'] - df['Stake']
    profit_diffs = np.abs(df['Profit'] - calc_profit)
    profit_errors = profit_diffs[profit_diffs > epsilon]
    if len(profit_errors) > 0:
        print(f"  [FAIL] Profit Logic Mismatch in {len(profit_errors)} rows.")
        errors += 1
    else:
        print("  [PASS] Profit Logic matches.")
        
    # 4. Audit Flat Profit Logic
    # Expected Flat Profit = (Odds - 1) if Win else -1
    calc_flat = np.where(df['IsWinner'] == 1, df['CurrentOdds'] - 1, -1)
    flat_diffs = np.abs(df['FlatProfit'] - calc_flat)
    flat_errors = flat_diffs[flat_diffs > epsilon]
    if len(flat_errors) > 0:
        print(f"  [FAIL] Flat Profit Logic Mismatch in {len(flat_errors)} rows.")
        print(f"  Example Error: Expected {calc_flat[flat_errors.index[0]]} vs Got {df.loc[flat_errors.index[0], 'FlatProfit']}")
        errors += 1
    else:
        print("  [PASS] Flat Profit Logic matches.")
    
    # 5. Audit Aggregate Stats
    total_bets = len(df)
    total_wins = df['IsWinner'].sum()
    strike_rate = (total_wins / total_bets) * 100
    
    total_profit_inv = df['Profit'].sum()
    total_stake_inv = df['Stake'].sum()
    roi_inv = (total_profit_inv / total_stake_inv) * 100
    
    total_profit_flat = df['FlatProfit'].sum()
    total_stake_flat = total_bets # 1 unit per bet
    roi_flat = (total_profit_flat / total_stake_flat) * 100
    
    print("-" * 60)
    print(f"  Verified Aggregates:")
    print(f"    Bets: {total_bets}")
    print(f"    Wins: {total_wins}")
    print(f"    Strike Rate: {strike_rate:.2f}%")
    print(f"    Profit (Inv): {total_profit_inv:.2f}u")
    print(f"    ROI (Inv): {roi_inv:.2f}%")
    print(f"    Profit (Flat): {total_profit_flat:.2f}u")
    print(f"    ROI (Flat): {roi_flat:.2f}%")
    print("-" * 60)
    
    if errors == 0:
        print("  [SUCCESS] All calculations verified correct.")
    else:
        print(f"  [FAILURE] Found {errors} logic discrepancies.")

if __name__ == "__main__":
    audit_file('backtest_longterm_leader.csv')
    audit_file('backtest_longterm_top3.csv')
