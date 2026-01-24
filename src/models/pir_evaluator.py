"""
PIR (Position In Run) Prediction Model - Fast Version with Progress Updates
Predicts FirstSplitPosition using historical data
"""

import sqlite3
import pandas as pd
import numpy as np
import time
import sys

DB_PATH = 'greyhound_racing.db'

def print_progress(msg):
    """Print with flush for immediate display"""
    print(msg)
    sys.stdout.flush()


class PIRModelEvaluator:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.output_buffer = []

    def log(self, msg):
        """Log to buffer and valid progress callback"""
        self.output_buffer.append(str(msg))
        print(msg) # Keep console output for debug
        # if self.callback: self.callback(msg) (Future extension)

    def evaluate(self):
        """Run full evaluation and return report string"""
        self.output_buffer = []
        self.log("="*70)
        self.log("PIR PREDICTION MODEL - FAST EVALUATION")
        self.log("="*70)
        
        conn = sqlite3.connect(self.db_path)
        
        # Step 1: Load data
        self.log("\n[1/4] Loading data...")
        start = time.time()
        
        query = """
        SELECT 
            ge.GreyhoundID, ge.RaceID, ge.Split, ge.Box,
            ge.SplitBenchmarkLengths, rm.MeetingDate
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.Split IS NOT NULL
          AND rm.MeetingDate >= '2023-01-01'
        ORDER BY ge.GreyhoundID, rm.MeetingDate
        """
        
        df = pd.read_sql_query(query, conn)
        self.log(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")
        
        # Step 2: Calculate historical averages
        self.log("\n[2/4] Calculating historical split averages...")
        start = time.time()
        
        df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
        df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
        df = df.sort_values(['GreyhoundID', 'MeetingDate']).reset_index(drop=True)
        
        # Rolling average last 5
        df['HistAvgSplit'] = df.groupby('GreyhoundID')['Split'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=3).mean()
        )
        
        df['PriorRaces'] = df.groupby('GreyhoundID').cumcount()
        
        self.log(f"  Done in {time.time()-start:.1f}s")
        
        # Step 3: Filter
        self.log("\n[3/4] Filtering and scoring...")
        start = time.time()
        
        df_valid = df[df['PriorRaces'] >= 5].copy()
        self.log(f"  {len(df_valid):,} entries with 5+ prior races")
        
        df_valid['Pred_HistAvg'] = df_valid['HistAvgSplit'].round().clip(1, 8)
        
        # Box adjustment
        box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
        df_valid['BoxAdj'] = df_valid['Box'].map(box_adj).fillna(0)
        df_valid['Pred_WithBox'] = (df_valid['HistAvgSplit'] + df_valid['BoxAdj']).round().clip(1, 8)
        
        self.log(f"  Done in {time.time()-start:.1f}s")
        
        # Step 4: Evaluate
        self.log("\n[4/4] Evaluating prediction accuracy...")
        
        self.log("\n" + "="*70)
        self.log("INDIVIDUAL DOG PREDICTION ACCURACY")
        self.log("="*70)
        
        for name, pred_col in [("Historical Avg", "Pred_HistAvg"), ("+ Box Adjust", "Pred_WithBox")]:
            valid = df_valid[df_valid[pred_col].notna()]
            error = abs(valid[pred_col] - valid['Split'])
            exact = (valid[pred_col] == valid['Split']).mean() * 100
            mae = error.mean()
            within1 = (error <= 1).mean() * 100
            
            self.log(f"\n{name}:")
            self.log(f"  Exact match: {exact:.1f}%")
            self.log(f"  Within 1 pos: {within1:.1f}%")
            self.log(f"  Mean abs error: {mae:.2f}")
            
        # Race Level
        self.log("\n" + "="*70)
        self.log("RACE-LEVEL: PREDICTING SPLIT LEADER")
        self.log("="*70)
        
        races = df_valid.groupby('RaceID')
        valid_races = 0
        correct_box = 0
        
        total_races = len(races)
        processed = 0
        
        for race_id, race_df in races:
            processed += 1
            if len(race_df) < 6: continue
            
            valid_races += 1
            actual_leader = race_df.loc[race_df['Split'].idxmin(), 'GreyhoundID']
            pred_leader_box = race_df.loc[(race_df['HistAvgSplit'] + race_df['BoxAdj']).idxmin(), 'GreyhoundID']
            
            if pred_leader_box == actual_leader:
                correct_box += 1

        self.log(f"\nAnalyzed {valid_races:,} valid races (6+ runners)")
        self.log(f"  Accuracy (+ Box Adj): {correct_box}/{valid_races} = {correct_box/valid_races*100:.1f}%")
        self.log(f"  Random Baseline:      12.5% (1 in 8)")
        
        conn.close()
        return "\n".join(self.output_buffer)

if __name__ == "__main__":
    evaluator = PIRModelEvaluator()
    print(evaluator.evaluate())
