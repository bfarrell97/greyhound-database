
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from autogluon.tabular import TabularPredictor
import os

# Configuration
DB_PATH = 'greyhound_racing.db'
MODEL_PATH = 'models/autogluon_margin_v25'
TRAIN_ROWS = 2000000

# Top features from V12 (retained) + New Margin Features
FEATURES_TO_USE = [
    'TrackBoxWinRate', 'DistAvgPos', 'Weight_x_Distance', 'BloodlineVsDog', 'TrackAvgPos',
    'TrainerTrackWinRate', 'BeyerStd', 'DaysSinceRace', 'WeightChange', 'SplitBest',
    'TimeQ25', 'SplitLag1', 'TimeBestRecent3', 'ExperiencePerAge', 'Age_x_Experience',
    'CareerStarts', 'CareerWinRate', 'DistExperience', 'TierExperience', 'TrainerDistWinRate',
    'SplitStd', 'TimeLag1', 'TrainerWinRate', 'LastWonDaysAgo', 'SplitAvg',
    'TrainerTrackRuns', 'TrainerFormVsAll', 'DamRuns', 'TimeStd', 'Trainer_x_Track',
    'DistWinRate', 'RaceFrequency30d', 'Time_x_Trainer', 'TrackExperience', 'TrainerStarts60d',
    'TimeAvg3', 'SpecialistScore', 'WeightStd', 'WeightAvg', 'SireWinRate',
    'Box', 'BoxPreference', 'TimeLag2', 'TimeTrend3', 'BeyerLag1',
    'CareerPlaces', 'PosAvg', 'BloodlineScore', 'TrainerWinRate60d', 'Time_x_TrainerForm',
    'Beyer_x_Trainer', 'DamWinRate', 'ThisBoxWinRate', 'TrackPlaceRate', 'DistPlaceRate',
    'TimeBest', 'TimeImproving', 'TimeAvg', 'WinRate5', 'Bloodline_x_Age',
    'WinsPerAge', 'TimeWorst', 'PosImprovement', 'TimeLag3', 'TrainerWinRate30d',
    'RaceFrequency60d', 'Form_x_Trainer', 'Rest_x_Form', 'CareerWins',
    # NEW MARGIN FEATURES
    'LastMargin', 'MarginAvg', 'MarginTrend', 'MarginBest', 'MarginWorst', 'MarginStd'
]

METRO_TRACKS = {'Wentworth Park', 'Albion Park', 'Angle Park', 'Sandown Park', 'The Meadows', 'Cannington'}
PROVINCIAL_TRACKS = {'Richmond', 'Richmond Straight', 'Nowra', 'The Gardens', 'Bulli', 'Dapto', 'Maitland', 
                     'Goulburn', 'Ipswich', 'Q Straight', 'Q1 Lakeside', 'Q2 Parklands', 'Gawler',
                     'Ballarat', 'Bendigo', 'Geelong', 'Sale', 'Cranbourne', 'Warrnambool', 'Mandurah'}

def get_tier(track):
    if track in METRO_TRACKS: return 2
    elif track in PROVINCIAL_TRACKS: return 1
    return 0

def safe_div(a, b, default=0):
    return a / b if b != 0 else default

def main():
    print("="*70)
    print("V25 MARGIN MODEL (AutoML + V12 Features + Margin History)")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    
    print("Loading Benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    # Query includes MARGIN
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box, ge.Margin,
           ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
           ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
           ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
           g.SireID, g.DamID, g.DateOfBirth
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate BETWEEN '2020-01-01' AND '2025-12-31'
      AND ge.Position IS NOT NULL
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Preprocessing (V12 Style)
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce').fillna(99) # Fill NaN margin with high value
    
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    
    # CRITICAL FIX: Winners have Margin=0 (DB stores margin to 2nd place)
    df.loc[df['Position'] == 1, 'Margin'] = 0.0
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    
    # LOOKUPS
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_recent = defaultdict(list)
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
    feature_rows = []
    processed = 0
    
    print(f"Processing {len(df):,} rows...")
    
    # --- FEATURE GENERATION LOOP ---
    for race_id, race_df in df.groupby('RaceID', sort=False):
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        # SKIP OLD DATA FOR TRAINING (Only generate history, start taking features > 2022?)
        # Or take all? V12 took all.
        
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            hist = dog_history.get(dog_id, [])
            
            # Predict only if history exists
            if len(hist) >= 3:
                recent = hist[-10:]
                # V12 Extractions
                times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                beyers = [h['beyer'] for h in recent if h['beyer'] is not None]
                positions = [h['position'] for h in recent if h['position'] is not None]
                margins = [h['margin'] for h in recent if h['margin'] is not None]
                
                if len(times) >= 3:
                    f = {'RaceID': race_id, 'Margin': r['Margin'], 'Date': race_date}
                    
                    # --- V12 Features (Re-implemented Inline) ---
                    f['TimeBest'] = min(times)
                    f['TimeAvg'] = np.mean(times)
                    f['TimeAvg3'] = np.mean(times[-3:])
                    f['TimeLag1'] = times[-1]
                    f['TimeStd'] = np.std(times)
                    f['TimeTrend3'] = times[-1] - times[-3] if len(times) >= 3 else 0
                    f['TimeBestRecent3'] = min(times[-3:])
                    f['TimeQ25'] = np.percentile(times, 25)
                    
                    # Split
                    splits = [h['split'] for h in recent if h['split'] is not None]
                    f['SplitBest'] = min(splits) if splits else 0
                    f['SplitAvg'] = np.mean(splits) if splits else 0
                    f['SplitLag1'] = splits[-1] if splits else 0
                    f['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                    
                    # Beyer
                    f['BeyerLag1'] = beyers[-1] if beyers else 77
                    f['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                    f['Beyer_x_Trainer'] = f['BeyerLag1'] # simplified
                    
                    # Pos
                    f['PosAvg'] = np.mean(positions)
                    f['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                    f['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                    f['CareerStarts'] = len(hist)
                    f['CareerWinRate'] = safe_div(f['CareerWins'], f['CareerStarts'])
                    f['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                    
                    # --- NEW MARGIN FEATURES ---
                    f['LastMargin'] = margins[-1] if margins else 10
                    f['MarginAvg'] = np.mean(margins) if margins else 10
                    f['MarginBest'] = min(margins) if margins else 0
                    f['MarginWorst'] = max(margins) if margins else 20
                    f['MarginStd'] = np.std(margins) if len(margins) >=3 else 0
                    f['MarginTrend'] = margins[-1] - np.mean(margins[-3:]) if len(margins) >= 3 else 0
                    
                    # Trainer
                    tid = r['TrainerID']
                    t_all = trainer_all[tid]
                    f['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'])
                    t_track = trainer_track[tid][track_id]
                    f['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'])
                    f['TrainerTrackRuns'] = min(t_track['runs'], 100) / 100
                    
                    # Track
                    track_runs = [h for h in hist if h['track_id'] == track_id]
                    f['TrackExperience'] = len(track_runs)
                    f['TrackAvgPos'] = np.mean([h['position'] for h in track_runs]) if track_runs else f['PosAvg']
                    f['TrackPlaceRate'] = safe_div(sum(1 for h in track_runs if h['position']<=3), len(track_runs))
                    
                    # Dist
                    dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                    f['DistExperience'] = len(dist_runs)
                    f['DistAvgPos'] = np.mean([h['position'] for h in dist_runs]) if dist_runs else f['PosAvg']
                    f['DistWinRate'] = safe_div(sum(1 for h in dist_runs if h['position']==1), len(dist_runs))
                    f['DistPlaceRate'] = safe_div(sum(1 for h in dist_runs if h['position']<=3), len(dist_runs))
                    
                    # Box
                    box = int(r['Box'])
                    f['Box'] = box
                    tb = track_box_wins[track_id][box]
                    f['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'])
                    this_box_runs = [h for h in hist if h['box'] == box]
                    f['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position']==1), len(this_box_runs))
                    f['BoxPreference'] = 0 # Simplified
                    
                    # Age/Rest
                    age_months = r['AgeMonths']
                    f['ExperiencePerAge'] = f['CareerStarts'] / (age_months + 1)
                    f['WinsPerAge'] = f['CareerWins'] / (age_months + 1)
                    days_since = (race_date - hist[-1]['date']).days
                    f['DaysSinceRace'] = days_since
                    f['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                    f['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                    f['LastWonDaysAgo'] = 999 
                    
                    # Bloodline (Simplified)
                    f['SireWinRate'] = safe_div(sire_stats[r['SireID']]['wins'], sire_stats[r['SireID']]['runs'])
                    f['DamWinRate'] = safe_div(dam_stats[r['DamID']]['wins'], dam_stats[r['DamID']]['runs'])
                    f['BloodlineScore'] = (f['SireWinRate'] + f['DamWinRate'])/2
                    f['BloodlineVsDog'] = f['BloodlineScore'] - f['CareerWinRate']
                    f['Bloodline_x_Age'] = f['BloodlineScore'] * age_months
                    f['DamRuns'] = dam_stats[r['DamID']]['runs']
                    
                    # Other
                    f['TierExperience'] = sum(1 for h in hist if h['tier'] == tier)
                    f['WeightAvg'] = np.mean([h['weight'] for h in recent if h['weight']]) if recent else 30
                    f['WeightChange'] = r['Weight'] - f['WeightAvg']
                    f['WeightStd'] = np.std([h['weight'] for h in recent if h['weight']]) if recent else 0
                    f['Weight_x_Distance'] = f['WeightChange'] * distance
                    
                    # Interactions
                    f['Time_x_Trainer'] = f['TimeBest'] * f['TrainerWinRate']
                    f['Time_x_TrainerForm'] = f['TimeBest'] * f['TrainerWinRate'] # approx
                    f['TrainerDistWinRate'] = f['TrainerWinRate'] # approx
                    f['TrainerStarts60d'] = 10 # approx
                    f['SpecialistScore'] = (f['DistWinRate'] + f['ThisBoxWinRate'])/2
                    f['TimeImproving'] = f['TimeTrend3']
                    f['PosImprovement'] = 0
                    f['Form_x_Trainer'] = 0
                    f['Rest_x_Form'] = 0
                    f['Trainer_x_Track'] = f['TrainerWinRate'] * f['TrainerTrackWinRate']
                    f['TrainerFormVsAll'] = 0
                    f['TrainerWinRate30d'] = f['TrainerWinRate']
                    f['TrainerWinRate60d'] = f['TrainerWinRate']
                    f['Age_x_Experience'] = age_months * f['CareerStarts']
                    
                    feature_rows.append(f)
            
            # Update History
            item = {
                'date': race_date, 'norm_time': r['NormTime'], 'beyer': r['BeyerSpeedFigure'],
                'position': r['Position'], 'track_id': track_id, 'distance': distance, 'tier': tier,
                'weight': r['Weight'], 'box': int(r['Box']), 'split': r['Split'],
                'margin': r['Margin']
            }
            dog_history[dog_id].append(item)
            
            # Stats Updates
            if r['Won']: 
                trainer_all[r['TrainerID']]['wins'] += 1
                trainer_track[r['TrainerID']][track_id]['wins'] += 1
                sire_stats[r['SireID']]['wins'] += 1
                dam_stats[r['DamID']]['wins'] += 1
                track_box_wins[track_id][int(r['Box'])]['wins'] += 1
            
            trainer_all[r['TrainerID']]['runs'] += 1
            trainer_track[r['TrainerID']][track_id]['runs'] += 1
            sire_stats[r['SireID']]['runs'] += 1
            dam_stats[r['DamID']]['runs'] += 1
            track_box_wins[track_id][int(r['Box'])]['runs'] += 1
            
        processed += 1
        if processed % 10000 == 0: print(f"Processed {processed} races...")

    # DataFrame
    df_feat = pd.DataFrame(feature_rows)
    print(f"Features Generated: {len(df_feat):,}")
    
    # Train/Test Split
    train_data = df_feat[df_feat['Date'] < '2024-06-01']
    test_data = df_feat[df_feat['Date'] >= '2024-06-01']
    
    cols = [c for c in FEATURES_TO_USE if c in df_feat.columns]
    print(f"Training on {len(cols)} features...")
    
    # AutoGluon
    print("Training AutoGluon...")
    predictor = TabularPredictor(label='Margin', path=MODEL_PATH, eval_metric='mean_absolute_error').fit(
        train_data[cols + ['Margin']],
        time_limit=600,
        presets='medium_quality'
    )
    
    # Eval
    perf = predictor.evaluate(test_data[cols + ['Margin']])
    print("Evaluation Results:", perf)
    
    # Feature Importance
    try:
        fi = predictor.feature_importance(test_data[cols + ['Margin']].sample(2000))
        print(fi.head(10))
    except: pass

    # --- VALUE BETTING SIMULATION ---
    print("\n" + "="*70)
    print("MARGIN-BASED VALUE BETTING")
    print("="*70)
    
    # 1. Add Predictions
    test_data['PredMargin'] = predictor.predict(test_data[cols])
    
    # 2. Convert to Probability (Softmin) per Race
    # P_i = exp(-PredMargin_i) / Sum(exp(-PredMargin_j))
    # Using a temperature scaling factor can help calibrate. Start with T=1.
    def calculate_probs(group):
        # Invert margin (lower is better). 
        # Shift scores to avoid overflow? No, margins are usually 0-20.
        # exp(-0) = 1, exp(-20) = small.
        scores = np.exp(-1.0 * group['PredMargin'])
        return scores / scores.sum()

    print("Calculating Predicted Probabilities & Rated Prices...")
    # Group by RaceID
    # We need RaceID in test_data?
    # Yes, it's in df_feat, and test_data is a slice.
    
    # Needs to be careful with index alignment
    test_data['RaceID'] = test_data['RaceID'].astype(int)
    
    probs = test_data.groupby('RaceID').apply(calculate_probs).reset_index(level=0, drop=True)
    test_data['PredProb'] = probs
    test_data['RatedPrice'] = 1 / test_data['PredProb']
    
    # Compare Rated Price vs BSP (Accuracy)
    print("\n" + "="*70)
    print("ACCURACY REPORT")
    print("="*70)

    # 1. Margin Accuracy
    test_data['PredMargin'] = predictor.predict(test_data[cols])
    margin_mae = np.mean(np.abs(test_data['Margin'] - test_data['PredMargin']))
    
    # Margin MAPE (Exclude Winners where Margin=0 to avoid inf)
    non_winners = test_data[test_data['Margin'] > 0.01].copy()
    if len(non_winners) > 0:
        margin_mape = np.mean(np.abs((non_winners['Margin'] - non_winners['PredMargin']) / non_winners['Margin'])) * 100
    else:
        margin_mape = 0
        
    print(f"MARGIN ACCURACY:")
    print(f"  MAE:  {margin_mae:.3f} lengths")
    print(f"  MAPE: {margin_mape:.2f}% (Non-Winners)")

    # 2. Price Accuracy (Rated Price vs BSP)
    print("\nPRICE ACCURACY (Rated Price vs BSP):")

    # Filter valid BSP for fair comparison
    valid = test_data[(test_data['BSP'] > 1) & (test_data['BSP'] < 50)].copy()
    
    # Error Metrics
    valid['Error'] = np.abs(valid['RatedPrice'] - valid['BSP'])
    valid['PctError'] = valid['Error'] / valid['BSP']
    
    mape = valid['PctError'].mean() * 100
    corr = valid[['RatedPrice', 'BSP']].corr().iloc[0,1]
    
    print(f"  Evaluated on {len(valid):,} rows (BSP < 50)")
    print(f"  MAPE (Rated vs BSP): {mape:.2f}%")
    print(f"Correlation: {corr:.4f}")
    
    # Breakdown by Odds Range
    print("\nAccuracy by BSP Range:")
    ranges = [(0,5), (5,10), (10,20), (20,50)]
    for low, high in ranges:
        subset = valid[(valid['BSP'] >= low) & (valid['BSP'] < high)]
        if len(subset) > 0:
            sub_mape = subset['PctError'].mean() * 100
            print(f"  ${low}-${high}: MAPE {sub_mape:.2f}% ({len(subset)} rows)")

if __name__ == "__main__":
    main()
