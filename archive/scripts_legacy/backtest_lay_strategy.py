"""
Backtest Lay Strategy (The False Favorite)
Goal: Capitalize on the -26% ROI of "Dominator" predictions by Laying them.
Market: Betfair Exchange (Simulated)
Commission: 5% on Net Winnings per market (Simplified to 5% per bet here for conservative est)
"""
import pandas as pd
import numpy as np

def backtest_lay():
    print("Loading Predictions...")
    try:
        df = pd.read_csv('safe_predictions.csv')
    except FileNotFoundError:
        print("Error: safe_predictions.csv not found.")
        return

    print(f"Loaded {len(df)} rows.")
    
    # ---------------------------------------------------------
    # RE-CREATE MARGIN LOGIC (Robus Rank 1 vs Rank 2)
    # ---------------------------------------------------------
    # Rank per RaceID
    df['PredRank'] = df.groupby('RaceID')['PredOverall'].rank(method='min')
    
    # Identify Rank 1s (Candidates)
    rank1s = df[df['PredRank'] == 1].copy()
    
    # Identify Rank 2s (Benchmark)
    rank2s = df[df['PredRank'] == 2][['RaceID', 'PredOverall']].copy()
    rank2s.columns = ['RaceID', 'Time2nd']
    
    # Merge Phase
    candidates = rank1s.merge(rank2s, on='RaceID', how='left')
    candidates['Margin'] = candidates['Time2nd'] - candidates['PredOverall']
    
    # DEBUG
    print(f"DEBUG: Candidates (Rank1)={len(candidates)}")
    print(f"DEBUG: Margins > 0.1={(candidates['Margin'] > 0.1).sum()}")
    
    # Use 'candidates' as our working dataframe for Lays
    df = candidates
    
    # ---------------------------------------------------------
    # LAY STRATEGY LOOP
    # ---------------------------------------------------------
    # Filters to test
    margins = [0.1, 0.2]
    max_odds_list = [4.0, 6.0, 8.0, 10.0] # Don't lay huge prices
    
    results = []
    
    STAKE = 100 # We confirm to win $100 (Liability varies)
    COMMISSION = 0.05
    
    print("\nLAY STRATEGY RESULTS (Target: Win $100 per Lay)")
    print(f"{'Config':<30} {'Lays':<6} {'Strike(L)%':<10} {'Liability':<10} {'Profit':<10} {'ROI%':<6}")
    print("-" * 90)
    
    for marg in margins:
        for max_odds in max_odds_list:
            # Filter: Model Likes it (Rank 1, Big Margin) AND Price is Low enough to Lay
            candidates = df[
                (df['PredRank'] == 1) & 
                (df['Margin'] > marg) & 
                (df['Odds'] >= 1.50) & # Don't lay unbackable favorites (<1.50 is dangerous/rare)
                (df['Odds'] <= max_odds)
            ].copy()
            
            if len(candidates) < 50: continue
            
            # Outcome Logic
            # Win for Layer -> Dog LOSES (IsWin == 0)
            # Loss for Layer -> Dog WINS (IsWin == 1)
            
            n_lays = len(candidates)
            layer_wins = (candidates['IsWin'] == 0).sum()
            layer_losses = (candidates['IsWin'] == 1).sum()
            
            # Money
            # We target $100 profit per bet.
            # Win: +$100 * (1 - Comm) = $95
            total_winnings = layer_wins * STAKE * (1 - COMMISSION)
            
            # Loss: We pay out (Odds - 1) * Stake
            # e.g. Lay $4.00 shot. Liability = $300.
            candidates['Liability'] = (candidates['Odds'] - 1) * STAKE
            total_losses = candidates[candidates['IsWin'] == 1]['Liability'].sum()
            
            net_profit = total_winnings - total_losses
            total_risk = candidates['Liability'].sum() # Total amount risked
            
            roi = net_profit / total_risk * 100 # ROI on Liabilty (Standard for layers)
            
            strike_rate = layer_wins / n_lays * 100
            
            config_name = f"Mg>{marg} Odds<{max_odds}"
            print(f"{config_name:<30} {n_lays:<6} {strike_rate:<10.1f} ${total_risk/1000:<8.1f}k ${net_profit:<9.1f} {roi:<6.1f}")
            
            results.append({
                'Config': config_name,
                'Lays': n_lays,
                'Profit': net_profit,
                'ROI': roi
            })

    # Best Result
    if results:
        best = sorted(results, key=lambda x: x['Profit'], reverse=True)[0]
        print("\nBest Lay Strategy:")
        print(f"{best['Config']} -> Profit: ${best['Profit']:.2f}")

if __name__ == "__main__":
    backtest_lay()
