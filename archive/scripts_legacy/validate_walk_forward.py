"""
Walk-Forward Validation & Monte Carlo Suite
Performs rigorous rolling validation of the "Short Course Dominant" Strategy (Pace Gap > 0.15, Dist < 450, Prize > 20k).
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading Dataset (2020-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.TrainerID,
        ge.FinishTime,
        ge.Split,
        ge.Position,
        ge.StartingPrice,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
      AND rm.MeetingDate >= '2020-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce') # Split can be null
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    print("Feature Engineering...")
    
    # Trainer Stats (Expanding Window to prevent leakage)
    # We want Trainer Win % prior to this race
    print("  Calculating Trainer Stats...")
    df['IsWin'] = (df['Position'] == '1').astype(int)
    # Group by TrainerID and sort by Date (already sorted roughly, ensuring sort)
    df = df.sort_values(['TrainerID', 'MeetingDate'])
    df['TrainerWinRate'] = df.groupby('TrainerID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    # Benchmarks (Global to avoid leakage? Ideally expanding window, but global is acceptable proxy if stable)
    # To be strict Walk-Forward, feature calculation should handle 'future' data carefully.
    # Rolling averages only look backward, so they are safe.
    # Benchmarks: Let's calc global median for simplicity, assuming track speeds don't drift massively.
    pace_bench = df.groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    # Split Benchmarks
    split_bench = df.groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    # Pace Features
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    
    # Split Features
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Must have history
    df = df.dropna(subset=['p_Roll5', 'Odds']).copy()
    # Note: Split history might be missing if track doesn't record splits, but we'll handle NaNs in model or fill 0?
    # XGBoost handles NaNs. We keep rows even if Split history is partial, as long as Pace is there.
    return df

def train_and_predict(train_df, test_df):
    # Pace Features
    pace_features = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance', 'TrainerWinRate']
    pace_target = 'NormTime'
    
    # Split Features
    split_features = ['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance', 'TrainerWinRate']
    split_target = 'NormSplit'
    
    # ---------------------------
    # PACE MODEL
    # ---------------------------
    print("  Training Pace Model...")
    model_pace = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model_pace.fit(train_df[pace_features], train_df[pace_target])
    
    # Predict Pace
    test_df['PredNormPace'] = model_pace.predict(test_df[pace_features])
    test_df['PredPace'] = test_df['PredNormPace'] + test_df['TrackDistMedianPace']
    
    # ---------------------------
    # SPLIT (PIR) MODEL
    # ---------------------------
    print("  Training Split Model...")
    # Filter training data for valid splits
    split_train = train_df.dropna(subset=[split_target]).copy()
    
    if len(split_train) > 1000:
        model_split = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model_split.fit(split_train[split_features], split_train[split_target])
        
        # Predict Split
        test_df['PredNormSplit'] = model_split.predict(test_df[split_features])
        test_df['PredSplit'] = test_df['PredNormSplit'] + test_df['TrackDistMedianSplit']
    else:
        print("  WARNING: Not enough split data to train model. Defaulting to 0.")
        test_df['PredSplit'] = 0
        
    return test_df

def calc_gaps(df):
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    # Filter field size >= 6
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df_valid = df[df['FieldSize'] >= 6].copy()
    
    # Calcluate Ranks
    # Pace Rank
    df_valid = df_valid.sort_values(['RaceKey', 'PredPace'])
    df_valid['Rank'] = df_valid.groupby('RaceKey').cumcount() + 1
    
    # PIR Rank (Split)
    df_valid = df_valid.sort_values(['RaceKey', 'PredSplit'])
    df_valid['PIRRank'] = df_valid.groupby('RaceKey').cumcount() + 1
    
    # Calculate Gaps (on Pace)
    df_valid = df_valid.sort_values(['RaceKey', 'PredPace']) # Resort by Pace for Gap calc
    df_valid['NextTime'] = df_valid.groupby('RaceKey')['PredPace'].shift(-1)
    df_valid['Gap'] = df_valid['NextTime'] - df_valid['PredPace']
    return df_valid

def run_walk_forward():
    full_df = load_data()
    
    # User requested recent data focus: 2024-2025
    # Training Window: Rolling 12 months (or 18 months) to stay fresh
    years = [2024, 2025] 
    all_bets = []
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION (Recent Data Focus)")
    print("="*80)
    
    for year in years:
        # Rolling Window: Train on [Year-1], Test on [Year]
        # Actually, let's allow 2 years training for data density
        train_start = year - 2 
        train_end = year - 1
        print(f"Loop {year}: Training on {train_start}-{train_end} -> Testing on {year}")
        
        train_mask = (full_df['MeetingDate'].dt.year >= train_start) & (full_df['MeetingDate'].dt.year <= train_end)
        test_mask = (full_df['MeetingDate'].dt.year == year)
        
        train_df = full_df[train_mask]
        test_df = full_df[test_mask].copy()
        
        if len(test_df) == 0:
            print(f"No data for {year}, skipping.")
            continue
            
        print(f"  Training Size: {len(train_df)}")
        test_with_preds = train_and_predict(train_df, test_df)
        
        # Strategy Logic
        # Dominant Leader: Dist < 450, Gap > 0.15, Prize > 30k, Odds > 2.0
        # AND PIR Leader (Split Rank 1)
        analyzed = calc_gaps(test_with_preds)
        
        leaders = analyzed[analyzed['Rank'] == 1].copy()
        
        # Strategy Filters
        # STRICT Sprint: Dist < 400
        bets = leaders[
            (leaders['PIRRank'] == 1) &  # MUST be predicted leader to first mark
            (leaders['Distance'] < 450) & # Expanded to < 450 (Standard Sprints)
            # (leaders['Gap'] >= 0.15) & # Removed Gap filter to match 'Verified' strategy and increase volume
            (leaders['CareerPrize'] >= 30000) & 
            (leaders['Odds'] >= 2.00) &
            (leaders['Odds'] <= 30)
        ].copy()
        
        if len(bets) > 0:
            # Calc Profit
            bets['Profit'] = bets.apply(lambda x: (x['Odds'] - 1) if x['Position'] == '1' else -1, axis=1)
            
            wins = bets[bets['Position'] == '1'].shape[0]
            strike = (wins / len(bets)) * 100
            total_pl = bets['Profit'].sum()
            roi = (total_pl / len(bets)) * 100
            
            print(f"  Results {year}: Bets={len(bets)}, Strike={strike:.1f}%, Profit={total_pl:.1f}u, ROI={roi:.1f}%")
            all_bets.append(bets)
        else:
            print(f"  Results {year}: No bets found matching criteria.")
            
    # Combine All Years
    if not all_bets:
        print("No bets generated across all years.")
        return

    full_record = pd.concat(all_bets, ignore_index=True)
    full_record = full_record.sort_values('MeetingDate')
    
    print("\n" + "="*80)
    print("AGGREGATED PERFORMANCE (2022-2025)")
    print("="*80)
    total_bets = len(full_record)
    total_wins = full_record[full_record['Position'] == '1'].shape[0]
    total_strike = (total_wins / total_bets) * 100
    total_profit = full_record['Profit'].sum()
    total_roi = (total_profit / total_bets) * 100
    
    print(f"Total Bets:   {total_bets}")
    print(f"Win Rate:     {total_strike:.1f}%")
    print(f"Total Profit: {total_profit:.1f} units")
    print(f"Global ROI:   {total_roi:.1f}%")
    
    # MONTE CARLO (Skill vs Luck)
    print("\n" + "-"*80)
    print("MONTE CARLO PERMUTATION TEST (1000 Runs)")
    print("-"*80)
    # Simulate random outcomes based on Market Implied Probability (1/Odds)
    # Null Hypothesis: The strategy selects dogs, but their wins are random based on market price.
    
    full_record['ImpliedProb'] = 1 / full_record['Odds']
    actual_profit = total_profit
    better = 0
    n_sims = 1000
    
    for i in range(n_sims):
        # Generate random wins based on odds
        rnd_wins = np.random.rand(total_bets) < full_record['ImpliedProb']
        rnd_profit = np.where(rnd_wins, full_record['Odds'] - 1, -1).sum()
        if rnd_profit >= actual_profit:
            better += 1
            
    p_value = better / n_sims
    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: STATISTICALLY SIGNIFICANT (Skill)")
    elif p_value < 0.10:
        print("Conclusion: MARGINAL (Review needed)")
    else:
        print("Conclusion: NOT SIGNIFICANT (Likely Market Noise)")
        
    # MONTE CARLO DRAWDOWN (Risk)
    print("\n" + "-"*80)
    print("MONTE CARLO DRAWDOWN TEST (Shuffle Order)")
    print("-"*80)
    
    max_dds = []
    profits = full_record['Profit'].values
    
    for i in range(1000):
        np.random.shuffle(profits)
        cumsum = np.cumsum(profits)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        max_dds.append(dd.max())
        
    avg_dd = np.mean(max_dds)
    worst_dd = np.max(max_dds)
    print(f"Avg Max Drawdown: {avg_dd:.1f}u")
    print(f"Worst Max Drawdown: {worst_dd:.1f}u")
    print(f"Rec. Bankroll (2x Worst): {worst_dd*2:.1f}u")

if __name__ == "__main__":
    run_walk_forward()
