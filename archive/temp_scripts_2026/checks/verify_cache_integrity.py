"""
Verify Cache Integrity script.
1. Load 365 days + Today's data.
2. Predict WITHOUT cache (Ground Truth).
3. Predict WITH cache (Test).
4. Compare Rolling Features and Probabilities.
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
try:
    from scripts.predict_v44_prod import MarketAlphaEngine
except ImportError:
    print("Could not import MarketAlphaEngine")
    sys.exit(1)

def verify_cache():
    print("--- VERIFYING CACHE INTEGRITY ---")
    
    # 1. Initialize Engine (Loads Cache)
    engine = MarketAlphaEngine()
    
    # 2. Get Today's Data
    print("Fetching Today's Data...")
    conn = sqlite3.connect('greyhound_racing.db')
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_history = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # We need full history for the NO-CACHE version to work correctly
    query = f"""
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
        ge.Split, ge.FinishTime, ge.Margin,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped, g.GreyhoundName as Dog, r.RaceTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '{start_history}'
    """
    df_full = pd.read_sql_query(query, conn)
    conn.close()
    
    # Sort for rolling calculation
    df_full = df_full.sort_values('MeetingDate')
    df_full['MeetingDate_Str'] = pd.to_datetime(df_full['MeetingDate']).dt.strftime('%Y-%m-%d')
    
    # Identify Today's rows (The ones we want to predict on)
    # Filter COMPLETED & VALID races for history (BSP > 0 AND Price5Min > 0) OR Today's races
    # This aligns with the Cache logic which filters both in SQL
    mask_history_valid = (df_full['BSP'] > 0) & (df_full['Price5Min'] > 0)
    mask_today_rows = (df_full['MeetingDate_Str'] == today_str)
    
    df_full = df_full[mask_history_valid | mask_today_rows].copy()
    
    # Identify Today's rows again after filtering
    mask_today = df_full['MeetingDate_Str'] == today_str
    today_indices = df_full[mask_today].index
    
    if len(today_indices) == 0:
        print("No races found for today to verify.")
        return

    print(f"Verifying on {len(today_indices)} runners today...")
    
    # 3. PREDICT (NO CACHE) - The "Ground Truth"
    # We pass the FULL DATAFRAME so rolling calc works on-the-fly
    print("\nRunning NO CACHE prediction (Computing from scratch)...")
    res_no_cache_full = engine.predict(df_full, use_cache=False)
    
    # Robustly identify today's rows in the OUTPUT (features might drop rows)
    if 'MeetingDate_Str' not in res_no_cache_full.columns:
        res_no_cache_full['MeetingDate_Str'] = pd.to_datetime(res_no_cache_full['MeetingDate']).dt.strftime('%Y-%m-%d')
    res_no_cache = res_no_cache_full[res_no_cache_full['MeetingDate_Str'] == today_str].copy().sort_values('EntryID')
    
    # 4. PREDICT (WITH CACHE)
    # We pass ONLY TODAY'S DATA (engine simulates history)
    print("Running WITH CACHE prediction (Simulating app usage)...")
    df_today = df_full[mask_today].copy()
    res_cache = engine.predict(df_today, use_cache=True).sort_values('EntryID')
    
    # 5. COMPARE (Merge on EntryID to ensure alignment)
    cols_to_check = [
        'Dog_Rolling_Steam_10', 'Trainer_Rolling_Steam_50',
        'Dog_Rolling_Drift_10', 'Trainer_Rolling_Drift_50',
        'Steam_Prob', 'Drift_Prob'
    ]
    print("\n--- COMPARISON RESULTS ---")
    
    # Ensure EntryID is available
    if 'EntryID' not in res_no_cache.columns: res_no_cache = res_no_cache.reset_index(drop=False)
    if 'EntryID' not in res_cache.columns: res_cache = res_cache.reset_index(drop=False)
    
    merged = pd.merge(
        res_no_cache[['EntryID', 'GreyhoundID', 'TrainerID'] + cols_to_check],
        res_cache[['EntryID'] + cols_to_check],
        on='EntryID',
        suffixes=('_NC', '_C'),
        how='inner'
    )
    
    print(f"Matched {len(merged)} rows for comparison.")
    
    for col in cols_to_check:
        nc_col = f"{col}_NC"
        c_col = f"{col}_C"
        
        if nc_col not in merged.columns or c_col not in merged.columns: continue
        
        diff = np.abs(merged[nc_col] - merged[c_col])
        max_diff = diff.max()
        
        status = "PASS" if max_diff < 1e-5 else "FAIL"
        print(f"{col:<30} | Max Diff: {max_diff:.6f} | Status: {status}")
        
        if status == "FAIL":
            print("   -> Sample Mismatches:")
            mismatches = merged[diff > 1e-5].head(3)
            for _, row in mismatches.iterrows():
                 print(f"      ID {row['EntryID']} (Dog {row['GreyhoundID']}): NoCache={row[nc_col]:.4f} vs Cache={row[c_col]:.4f} [TrainerID={row.get('TrainerID')}]")

if __name__ == "__main__":
    verify_cache()
