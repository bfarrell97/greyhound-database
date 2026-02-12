"""
BACKTEST: Final Custom Configuration
- Period: 2024-2025
- Thresholds: BACK >= 0.60, LAY >= 0.70 (NO TRAP)
- BACK Strategy: Odds $2-$30, 4% Target Profit
- LAY Strategy: Odds < $30, Max Liability 10% of Bank
- Compounding Bank
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys, os
sys.path.append('.')
from src.features.feature_engineering_v41 import FeatureEngineerV41
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def run_backtest():
    print("="*80)
    print("BACKTEST: FINAL CUSTOM CONFIGURATION")
    print("Period: 2024-2025")
    print("Settings:")
    print("  - BACK: Threshold >= 0.60 (No Trap), Odds $2.00 - $40.00")
    print("  - BACK Staking: 4% Target Profit (Compounding)")
    print("  - LAY: Threshold >= 0.70, Odds < $40.00")
    print("  - LAY Staking: Max Liability 10% of Bank (Compounding)")
    print("="*80)
    
    # 1. LOAD DATA
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        ge.Price5Min,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.GreyhoundName as Dog, g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2023-06-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 2. FEATURES
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    
    df_bt = df[
        (df['MeetingDate'] >= '2024-01-01') & 
        (df['MeetingDate'] <= '2025-12-31') & 
        (df['Price5Min'] > 0)
    ].sort_values('MeetingDate').copy()
    
    print(f"Population: {len(df_bt)} runs")

    # 3. MODELS
    model_v41 = joblib.load(MODEL_V41)
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)

    v41_cols = fe.get_feature_list()
    for c in v41_cols:
        if c not in df_bt.columns: df_bt[c] = 0
        df_bt[c] = pd.to_numeric(df_bt[c], errors='coerce').fillna(0)
        
    dmatrix = xgb.DMatrix(df_bt[v41_cols])
    df_bt['V41_Prob'] = model_v41.predict(dmatrix)
    df_bt['V41_Price'] = 1.0 / df_bt['V41_Prob']
    
    df_bt['Discrepancy'] = df_bt['Price5Min'] / df_bt['V41_Price']
    df_bt['Price_Diff'] = df_bt['Price5Min'] - df_bt['V41_Price']
    
    alpha_feats = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    
    df_bt['Steam_Prob'] = model_v42.predict_proba(df_bt[alpha_feats])[:, 1]
    df_bt['Drift_Prob'] = model_v43.predict_proba(df_bt[alpha_feats])[:, 1]
    
    # BACK Logic: Sliding Scale
    # Initialize PASS
    df_bt['Signal'] = 'PASS'
    
    # Back Candidates (Broad sweep for optimization, detail in loop)
    # We'll just mark anything with decent steam as potential BACK to enter loop
    df_bt.loc[df_bt['Steam_Prob'] >= 0.55, 'Signal'] = 'BACK'
    
    # LAY Logic: High Volume Standard (<4->0.55, 4-8->0.60, >8->0.65)
    cond_low = (df_bt['Price5Min'] < 4.0) & (df_bt['Drift_Prob'] >= 0.55)
    cond_mid = (df_bt['Price5Min'] >= 4.0) & (df_bt['Price5Min'] < 8.0) & (df_bt['Drift_Prob'] >= 0.60)
    cond_high = (df_bt['Price5Min'] >= 8.0) & (df_bt['Drift_Prob'] >= 0.65)
    
    df_bt.loc[cond_low | cond_mid | cond_high, 'Signal'] = 'LAY'
    
    df_bt['Position'] = pd.to_numeric(df_bt['Position'], errors='coerce').fillna(99)
    
    # 4. SIMULATION
    bets = []
    curr_bank = 200.0
    start_bank = 200.0
    
    # EXCLUSION RULE: Max 2 Lays per Race
    # Update mask to use Sliding Scale Logic
    # Price < 4 -> 0.55, Price < 8 -> 0.60, Else 0.65
    def get_threshold(p):
        if p < 4.0: return 0.55
        if p < 8.0: return 0.60
        return 0.65
        
    # Vectorized mask (High Volume Standard)
    cond_low = (df_bt['Price5Min'] < 4.0) & (df_bt['Drift_Prob'] >= 0.55)
    cond_mid = (df_bt['Price5Min'] >= 4.0) & (df_bt['Price5Min'] < 8.0) & (df_bt['Drift_Prob'] >= 0.60)
    cond_high = (df_bt['Price5Min'] >= 8.0) & (df_bt['Drift_Prob'] >= 0.65)
    
    # Updated Lay Condition: Odds < 20.0 (User Request)
    lay_mask = (cond_low | cond_mid | cond_high) & (df_bt['Price5Min'] < 20.0)
    
    lay_candidates = df_bt[lay_mask]
    race_lay_counts = lay_candidates['RaceID'].value_counts()
    
    print("\n--- Lay Bets per Race Distribution ---")
    dist = race_lay_counts.value_counts().sort_index()
    for count, frequency in dist.items():
        print(f"   {count} Lays: {frequency} races")
    print("--------------------------------------")
    
    excluded_lay_races = set(race_lay_counts[race_lay_counts > 2].index)
    print(f"\n[Rule] Excluding {len(excluded_lay_races)} races with > 2 Lay candidates.")

    print("\n--- Low Price Drifter Analysis ---")
    low_price = df_bt[df_bt['Price5Min'] < 8.0]
    print(f"Runners < $8.00: {len(low_price)}")
    print(f"  Prob 0.50-0.55: {len(low_price[(low_price['Drift_Prob'] >= 0.50) & (low_price['Drift_Prob'] < 0.55)])}")
    print(f"  Prob 0.55-0.60: {len(low_price[(low_price['Drift_Prob'] >= 0.55) & (low_price['Drift_Prob'] < 0.60)])}")
    print(f"  Prob 0.60-0.65: {len(low_price[(low_price['Drift_Prob'] >= 0.60) & (low_price['Drift_Prob'] < 0.65)])}")
    print(f"  Prob >= 0.65:   {len(low_price[low_price['Drift_Prob'] >= 0.65])}")
    print("----------------------------------\n")

    print("\nSimulating Bets...")
    
    # Initialize Stats Container for BSP Analysis
    lay_stats = {
        'Missed_BSP': {'bets': 0, 'pnl': 0.0, 'invested': 0.0, 'details': []},
        'Beat_BSP':   {'bets': 0, 'pnl': 0.0, 'invested': 0.0}
    }
    
    curr_bank = 2000.0
    total_pnl = 0.0
    bets = []
    logs = []
    
    debug_count = 0
    for idx, row in df_bt.iterrows():
        signal = row.get('Signal', 'PASS')
        
        curr_price = row['Price5Min']
        
        # DEBUG TRACE
        if row.get('Signal') == 'LAY' and debug_count < 10:
             print(f"[DEBUG] Row {idx}: Price={curr_price}, Drift={row['Drift_Prob']}, Signal={signal}")
             debug_count += 1
        # COMPLEX LOGIC
        
        # BACK STRATEGY - User Sliding Scale
        # $1-2: 0.60
        # $2-6: 0.55
        # $6-10: 0.60
        # $10-40: 0.70
        is_back = False
        bk_thresh = 0.99 # Default to ignore
        
        if curr_price < 2.0: bk_thresh = 0.60
        elif curr_price < 6.0: bk_thresh = 0.55
        elif curr_price < 10.0: bk_thresh = 0.60
        elif curr_price <= 40.0: bk_thresh = 0.70
        else: bk_thresh = 0.99 # Ignore > 40
        
        if row['Steam_Prob'] >= bk_thresh:
            is_back = True

        if is_back:
            signal = 'BACK'
            
        # LAY STRATEGY - High Volume Standard (Restored)
        # < $4 (@ 0.55), $4-$8 (@ 0.60), > $8 (@ 0.65)
        is_lay = False
        threshold = 0.65
        
        if curr_price < 4.00:
            threshold = 0.55
        elif curr_price < 8.00:
            threshold = 0.60
        else:
            threshold = 0.65
            
        if row['Drift_Prob'] >= threshold:
            if curr_price < 20.0:
                if row['RaceID'] not in excluded_lay_races:
                    is_lay = True
                
        if is_lay:
            signal = 'LAY'
            
        # Overwrite signal from dataframe with new logic
        if is_back: signal = 'BACK'
        elif is_lay: signal = 'LAY'
        else: signal = 'PASS'
        
        if signal == 'PASS': continue
            
        stake = 0.0
        risk = 0.0
        
        # DETERMINE EXECUTION PRICE
        # default to Price5Min
        exec_price = curr_price 
        
        if signal == 'LAY':
            # User Request: Use Price5Min (Default)
            exec_price = curr_price
            if not exec_price or exec_price <= 1.0:
                continue
        
        # STAKING (Compounding)
        if signal == 'BACK':
            # 4% Target Profit
            target = curr_bank * 0.04
            if exec_price > 1.01:
                stake = target / (exec_price - 1.0)
            risk = stake
        else:
            # 10% Liability Cap
            risk = curr_bank * 0.10
            if exec_price > 1.01:
                stake = risk / (exec_price - 1.0)
                
        # SAFETY: Don't bet more than we have
        if risk > curr_bank:
            risk = curr_bank
            if signal == 'BACK': stake = risk
            else: stake = risk / (exec_price - 1.0)

        # OUTCOME
        win = (row['Position'] == 1)
        pnl = 0.0
        
        if signal == 'BACK':
            if win:
                profit = stake * (exec_price - 1) * 0.95 # 5% Comm
                pnl = profit
            else:
                pnl = -stake
        else: # LAY
            if win:
                pnl = -risk
            else:
                pnl = stake * 0.95 # Commission
                    
        # BSP COMPARISON
        bsp = row['BSP'] if 'BSP' in row else 0.0 # Ensure BSP exists, default to 0.0 if not
        beat_bsp = False
        if signal == 'LAY':
             # For Lay, Beating BSP means Laying at LOWER odds than SP (Lower Liability)??
             # actually, if I Lay at $2.00 and SP is $2.50. I risked $1 to win $1. Handbrake.
             # If I waited for SP, I would risk $1.50 to win $1.
             # So Laying at $2.00 is BETTER (Less Liability for same reward).
             # So Price < BSP = BEAT BSP.
             if exec_price < bsp: beat_bsp = True
        elif signal == 'BACK':
             # For Back, Price > BSP = BEAT BSP.
             if exec_price > bsp: beat_bsp = True
             
        # Record Stats
        curr_bank += pnl
        total_pnl += pnl
        
        # Track Lays specifically for User Request
        if signal == 'LAY':
            is_missed = (exec_price > bsp)
            
            stats_key = 'Missed_BSP' if is_missed else 'Beat_BSP'
            
            # Global Stats
            lay_stats[stats_key]['bets'] += 1
            lay_stats[stats_key]['pnl'] += pnl
            lay_stats[stats_key]['invested'] += risk # Liability is the investment for Lay? Or Stake? ROI usually on Liability.
            
            if is_missed:
                # Store detailed missed bets for review
                lay_stats['Missed_BSP']['details'].append({
                    'Dog': row['Dog'] if 'Dog' in row else 'N/A', # Ensure Dog exists
                    'Track': row['TrackName'] if 'TrackName' in row else 'N/A', # Ensure TrackName exists
                    'Price': exec_price, 'BSP': bsp,
                    'Result': 'WIN' if win else 'LOSS', 'PnL': pnl
                })

        logs.append({
            'Date': row['MeetingDate'],
            'Dog': row['Dog'] if 'Dog' in row else 'N/A', # Ensure Dog exists
            'Signal': signal,
            'Price': exec_price,
            'BSP': bsp,
            'BeatBSP': beat_bsp,
            'Result': 'WIN' if win else 'LOSS',
            'PnL': pnl,
            'Bank': curr_bank
        })
        
        bets.append({
            'Date': row['MeetingDate'],
            'Dog': row['Dog'],
            'Signal': signal,
            'Price': exec_price,
            'BSP': bsp,
            'Stake': stake,
            'Risk': risk,
            'PnL': pnl,
            'Win': 1 if win else 0
        })
        
        # Prevent bankruptcy
        if curr_bank <= 5.0:
            print("❌ BANKRUPTCY!")
            break

    # 5. REPORT
    res = pd.DataFrame(bets)
    
    if res.empty:
        print("\nNo bets generated.")
        return
    
    print("\n" + "="*60)
    print("RESULTS: FINAL CONFIGURATION")
    print("="*60)
    print(f"Final Bank: ${curr_bank:.2f} (Start: ${start_bank:.2f})")
    print(f"Total Profit: ${curr_bank - start_bank:.2f}")
    
    print(f"\nTotal Bets: {len(res)}")
    
    for sig in ['BACK', 'LAY']:
        sub = res[res['Signal'] == sig]
        if sub.empty: continue
        stake = sub['Stake'].sum()
        risk = sub['Risk'].sum()
        pnl = sub['PnL'].sum()
        wins = sub['Win'].sum()
        roi = (pnl / risk * 100) if risk > 0 else 0
        print(f"  {sig}: {len(sub)} bets | Win Rate: {wins/len(sub)*100:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
        
    print(f"\n{'='*40}")
    
    res['Year'] = res['Date'].dt.year
    yearly = res.groupby('Year').agg({
        'PnL': 'sum', 
        'Risk': 'sum', 
        'Signal': 'count'
    }).rename(columns={'Signal': 'Bets'})
    yearly['ROI'] = yearly['PnL'] / yearly['Risk'] * 100
    print("\n--- By Year ---")
    print(yearly.to_string())
    
    # Report Lay BSP Stats (User Request)
    print("\n" + "="*60)
    print("LAY STRATEGY: BSP ANALYSIS (Did we beat market?)")
    print("="*60)
    
    # 1. Missed BSP (Price > BSP) -> Higher Liability than SP
    m_bets = lay_stats['Missed_BSP']['bets']
    m_pnl = lay_stats['Missed_BSP']['pnl']
    m_inv = lay_stats['Missed_BSP']['invested']
    m_roi = (m_pnl / m_inv * 100) if m_inv > 0 else 0.0
    
    print(f"\n[DID NOT BEAT BSP] (Lay Price > BSP)")
    print(f"  Bets: {m_bets}")
    print(f"  P&L:  ${m_pnl:.2f}")
    print(f"  ROI:  {m_roi:.2f}% (Liability Base)")
    
    # 2. Beat BSP (Price <= BSP) -> Lower Liability than SP
    b_bets = lay_stats['Beat_BSP']['bets']
    b_pnl = lay_stats['Beat_BSP']['pnl']
    b_inv = lay_stats['Beat_BSP']['invested']
    b_roi = (b_pnl / b_inv * 100) if b_inv > 0 else 0.0
    
    print(f"\n[BEAT BSP] (Lay Price <= BSP)")
    print(f"  Bets: {b_bets}")
    print(f"  P&L:  ${b_pnl:.2f}")
    print(f"  ROI:  {b_roi:.2f}%")
    
    # Combined
    tot_bets = m_bets + b_bets
    tot_pnl = m_pnl + b_pnl
    tot_inv = m_inv + b_inv
    tot_roi = (tot_pnl / tot_inv * 100) if tot_inv > 0 else 0.0
    
    print(f"\n[TOTAL LAYS]")
    print(f"  Bets: {tot_bets}")
    print(f"  P&L:  ${tot_pnl:.2f}")
    print(f"  ROI:  {tot_roi:.2f}%")
    
    res.to_csv("backtest_final_config.csv", index=False)
    print("\nSaved to backtest_final_config.csv")

if __name__ == "__main__":
    run_backtest()
