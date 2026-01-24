"""
PIR PREDICTION MODEL v2 - Optimized with progress updates
Predicts Position In Run (FirstSplitPosition) using historical data
"""

import sqlite3
import pandas as pd
import numpy as np
import time
import sys

DB_PATH = 'greyhound_racing.db'

def flush_print(msg):
    """Print with immediate flush"""
    print(msg)
    sys.stdout.flush()

def main():
    flush_print("="*70)
    flush_print("PIR PREDICTION MODEL v2")
    flush_print("="*70)
    flush_print("")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Step 1: Load data with historical averages pre-calculated
    flush_print("[1/3] Loading data with historical splits...")
    start = time.time()
    
    query = """
    WITH dog_history AS (
        SELECT 
            ge.GreyhoundID,
            ge.RaceID,
            ge.FirstSplitPosition,
            ge.Box,
            rm.MeetingDate,
            AVG(ge2.FirstSplitPosition) as HistAvgSplit,
            COUNT(ge2.FirstSplitPosition) as HistCount
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        LEFT JOIN Races r2 ON r2.RaceID != r.RaceID
        LEFT JOIN RaceMeetings rm2 ON r2.MeetingID = rm2.MeetingID AND rm2.MeetingDate < rm.MeetingDate
        LEFT JOIN GreyhoundEntries ge2 ON ge2.GreyhoundID = ge.GreyhoundID 
            AND ge2.RaceID = r2.RaceID 
            AND ge2.FirstSplitPosition IS NOT NULL
        WHERE ge.FirstSplitPosition IS NOT NULL
          AND rm.MeetingDate >= '2023-01-01'
        GROUP BY ge.GreyhoundID, ge.RaceID
        HAVING HistCount >= 5
    )
    SELECT * FROM dog_history
    LIMIT 200000
    """
    
    # This query is too slow, let's do it differently
    # Load raw data and compute in pandas
    
    query = """
    SELECT 
        ge.GreyhoundID, ge.RaceID, ge.FirstSplitPosition, ge.Box,
        rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.FirstSplitPosition IS NOT NULL
      AND rm.MeetingDate >= '2022-01-01'
    ORDER BY ge.GreyhoundID, rm.MeetingDate, r.RaceID
    """
    
    df = pd.read_sql_query(query, conn)
    flush_print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
    
    # Step 2: Calculate expanding historical average
    flush_print("")
    flush_print("[2/3] Computing historical averages...")
    start = time.time()
    
    # Sort and compute expanding mean (excluding current row)
    df = df.sort_values(['GreyhoundID', 'MeetingDate', 'RaceID'])
    
    # Cumsum trick for expanding average
    df['CumSplit'] = df.groupby('GreyhoundID')['FirstSplitPosition'].cumsum().shift(1)
    df['CumCount'] = df.groupby('GreyhoundID').cumcount()
    df['HistAvgSplit'] = df['CumSplit'] / df['CumCount']
    
    # Filter to those with 5+ prior races
    df = df[df['CumCount'] >= 5].copy()
    flush_print(f"  {len(df):,} entries with 5+ prior races")
    flush_print(f"  Done in {time.time()-start:.1f}s")
    
    # Step 3: Evaluate predictions
    flush_print("")
    flush_print("[3/3] Evaluating predictions...")
    flush_print("")
    
    # Individual accuracy
    df['PredSplit'] = df['HistAvgSplit'].round().clip(1, 8)
    df['Error'] = abs(df['PredSplit'] - df['FirstSplitPosition'])
    
    flush_print("="*70)
    flush_print("INDIVIDUAL PREDICTION ACCURACY")
    flush_print("="*70)
    flush_print(f"  Exact match: {(df['Error'] == 0).mean()*100:.1f}%")
    flush_print(f"  Within 1 pos: {(df['Error'] <= 1).mean()*100:.1f}%")
    flush_print(f"  Within 2 pos: {(df['Error'] <= 2).mean()*100:.1f}%")
    flush_print(f"  Mean abs error: {df['Error'].mean():.2f}")
    
    # Race-level: predict leader
    flush_print("")
    flush_print("="*70)
    flush_print("RACE-LEVEL PREDICTIONS")
    flush_print("="*70)
    flush_print("")
    
    # Add box-adjusted prediction
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
    df['PredWithBox'] = df['HistAvgSplit'] + df['BoxAdj']
    
    # Group by race and evaluate
    races = df.groupby('RaceID')
    race_ids = list(races.groups.keys())
    
    correct_hist = 0
    correct_box = 0
    correct_top2_hist = 0
    correct_top2_box = 0
    valid_races = 0
    
    flush_print(f"Analyzing {len(race_ids):,} races...")
    
    for i, race_id in enumerate(race_ids):
        if i > 0 and i % 20000 == 0:
            flush_print(f"  {i:,}/{len(race_ids):,} races processed...")
        
        race_df = races.get_group(race_id)
        if len(race_df) < 6:
            continue
        
        valid_races += 1
        
        # Actual leader
        actual_leader = race_df.loc[race_df['FirstSplitPosition'].idxmin(), 'GreyhoundID']
        actual_top2 = set(race_df.nsmallest(2, 'FirstSplitPosition')['GreyhoundID'])
        
        # Predicted leader (lowest HistAvgSplit)
        pred_leader_hist = race_df.loc[race_df['HistAvgSplit'].idxmin(), 'GreyhoundID']
        pred_leader_box = race_df.loc[race_df['PredWithBox'].idxmin(), 'GreyhoundID']
        
        # Predicted top 2
        pred_top2_hist = set(race_df.nsmallest(2, 'HistAvgSplit')['GreyhoundID'])
        pred_top2_box = set(race_df.nsmallest(2, 'PredWithBox')['GreyhoundID'])
        
        if pred_leader_hist == actual_leader:
            correct_hist += 1
        if pred_leader_box == actual_leader:
            correct_box += 1
        if len(actual_top2 & pred_top2_hist) >= 1:
            correct_top2_hist += 1
        if len(actual_top2 & pred_top2_box) >= 1:
            correct_top2_box += 1
    
    flush_print("")
    flush_print(f"Valid races: {valid_races:,}")
    flush_print("")
    flush_print("PREDICTING SPLIT LEADER:")
    flush_print(f"  Historical avg: {correct_hist}/{valid_races} = {correct_hist/valid_races*100:.1f}%")
    flush_print(f"  + Box adjust:   {correct_box}/{valid_races} = {correct_box/valid_races*100:.1f}%")
    flush_print(f"  Random baseline: 12.5%")
    flush_print("")
    flush_print("PREDICTING AT LEAST 1 OF TOP 2:")
    flush_print(f"  Historical avg: {correct_top2_hist}/{valid_races} = {correct_top2_hist/valid_races*100:.1f}%")
    flush_print(f"  + Box adjust:   {correct_top2_box}/{valid_races} = {correct_top2_box/valid_races*100:.1f}%")
    flush_print(f"  Random baseline: 25%")
    
    # Breakdown by confidence
    flush_print("")
    flush_print("="*70)
    flush_print("ACCURACY BY CONFIDENCE (gap to 2nd fastest)")
    flush_print("="*70)
    flush_print("")
    
    high_conf_correct = 0
    high_conf_total = 0
    med_conf_correct = 0
    med_conf_total = 0
    low_conf_correct = 0
    low_conf_total = 0
    
    for race_id in race_ids:
        race_df = races.get_group(race_id)
        if len(race_df) < 6:
            continue
        
        sorted_preds = race_df.sort_values('HistAvgSplit')
        gap = sorted_preds.iloc[1]['HistAvgSplit'] - sorted_preds.iloc[0]['HistAvgSplit']
        
        actual_leader = race_df.loc[race_df['FirstSplitPosition'].idxmin(), 'GreyhoundID']
        pred_leader = sorted_preds.iloc[0]['GreyhoundID']
        correct = pred_leader == actual_leader
        
        if gap >= 1.0:
            high_conf_total += 1
            if correct:
                high_conf_correct += 1
        elif gap >= 0.5:
            med_conf_total += 1
            if correct:
                med_conf_correct += 1
        else:
            low_conf_total += 1
            if correct:
                low_conf_correct += 1
    
    flush_print(f"High confidence (gap >= 1.0): {high_conf_correct}/{high_conf_total} = {high_conf_correct/high_conf_total*100:.1f}%")
    flush_print(f"Medium confidence (0.5-1.0): {med_conf_correct}/{med_conf_total} = {med_conf_correct/med_conf_total*100:.1f}%")
    flush_print(f"Low confidence (< 0.5):      {low_conf_correct}/{low_conf_total} = {low_conf_correct/low_conf_total*100:.1f}%")
    
    flush_print("")
    flush_print("="*70)
    flush_print("COMPLETE")
    flush_print("="*70)
    
    conn.close()

if __name__ == "__main__":
    main()
