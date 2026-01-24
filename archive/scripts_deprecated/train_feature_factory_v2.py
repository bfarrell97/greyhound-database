"""
Feature Factory V2 - Expanded Feature Generation & Selection
=============================================================
Starts with 69 proven features from V12, adds extensive feature
combinations to reach ~500 total features for importance ranking.

Feature Categories:
1. Core 69 features from Feature Factory V1
2. Extended feature interactions (200+)
3. Polynomial features (key combinations squared/cubed)
4. Ratio features
5. Difference features
6. Rank-based features
7. Rolling window variations
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

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

def train_feature_factory_v2():
    print("="*70)
    print("FEATURE FACTORY V2 - 500 CANDIDATE FEATURES")
    print("="*70)
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("\n[1/7] Loading benchmarks...")
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime FROM Benchmarks", conn)
    bench_lookup = {(r['TrackName'], r['Distance']): r['MedianTime'] for _, r in bench_df.iterrows()}
    
    print("\n[2/7] Loading data...")
    query = """
    SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Box,
           ge.FinishTime, ge.Split, ge.BeyerSpeedFigure, ge.Weight,
           ge.FirstSplitPosition, ge.SecondSplitTime, ge.SecondSplitPosition,
           ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
           g.SireID, g.DamID, g.DateOfBirth
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate BETWEEN '2019-01-01' AND '2025-11-30'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    ORDER BY rm.MeetingDate, ge.RaceID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df = df.dropna(subset=['Position'])
    df['Won'] = (df['Position'] == 1).astype(int)
    
    for col in ['FinishTime', 'Split', 'BeyerSpeedFigure', 'BSP', 'Weight',
                'FirstSplitPosition', 'SecondSplitTime', 'SecondSplitPosition']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Tier'] = df['TrackName'].apply(get_tier)
    df['Benchmark'] = df.apply(lambda r: bench_lookup.get((r['TrackName'], r['Distance']), np.nan), axis=1)
    df['NormTime'] = df['FinishTime'] - df['Benchmark']
    df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
    df['AgeMonths'] = ((df['MeetingDate'] - df['DateOfBirth']).dt.days / 30.44).fillna(30)
    df['ClosingTime'] = df['FinishTime'] - df['SecondSplitTime']
    df['PositionDelta'] = df['FirstSplitPosition'] - df['Position']
    
    print(f"Loaded {len(df):,} entries")
    
    print("\n[3/7] Building lookups...")
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_recent = defaultdict(list)
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
    print("\n[4/7] Building 500 features...")
    feature_rows = []
    processed = 0
    
    for race_id, race_df in df.groupby('RaceID', sort=False):
        if len(race_df) < 4:
            continue
        
        race_date = race_df['MeetingDate'].iloc[0]
        distance = race_df['Distance'].iloc[0]
        track_id = race_df['TrackID'].iloc[0]
        track_name = race_df['TrackName'].iloc[0]
        tier = race_df['Tier'].iloc[0]
        
        if race_date >= datetime(2024, 1, 1):
            for _, r in race_df.iterrows():
                dog_id = r['GreyhoundID']
                hist = dog_history.get(dog_id, [])
                
                if len(hist) >= 3:
                    recent = hist[-10:]
                    
                    times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    beyers = [h['beyer'] for h in recent if h['beyer'] is not None]
                    positions = [h['position'] for h in recent if h['position'] is not None]
                    closings = [h['closing'] for h in recent if h['closing'] is not None]
                    deltas = [h['pos_delta'] for h in recent if h['pos_delta'] is not None]
                    weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
                    splits = [h['split'] for h in recent if h['split'] is not None]
                    
                    if len(times) >= 3:
                        features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Tier': tier}
                        
                        # =====================================================
                        # SECTION 1: CORE 69 FEATURES FROM V1
                        # =====================================================
                        
                        # Time features (12)
                        features['TimeBest'] = min(times)
                        features['TimeWorst'] = max(times)
                        features['TimeAvg'] = np.mean(times)
                        features['TimeAvg3'] = np.mean(times[-3:])
                        features['TimeLag1'] = times[-1]
                        features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                        features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                        features['TimeStd'] = np.std(times) if len(times) >= 3 else 0
                        features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0
                        features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                        features['TimeBestRecent3'] = min(times[-3:])
                        features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                        
                        # Split features (4)
                        features['SplitBest'] = min(splits) if splits else 0
                        features['SplitAvg'] = np.mean(splits) if splits else 0
                        features['SplitLag1'] = splits[-1] if splits else 0
                        features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                        
                        # Beyer features (2)
                        features['BeyerLag1'] = beyers[-1] if beyers else 77
                        features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                        
                        # Position/Win features (10)
                        features['PosAvg'] = np.mean(positions)
                        features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                        features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                        features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                        features['CareerStarts'] = min(len(hist), 100)
                        features['CareerWinRate'] = safe_div(features['CareerWins'], features['CareerStarts'], 0.12)
                        features['LastWonDaysAgo'] = 999
                        for i, h in enumerate(reversed(hist)):
                            if h['position'] == 1:
                                features['LastWonDaysAgo'] = (race_date - h['date']).days
                                break
                        
                        # Form trend
                        if len(positions) >= 5:
                            first_half = np.mean(positions[:len(positions)//2])
                            second_half = np.mean(positions[len(positions)//2:])
                            form_trend = first_half - second_half
                        else:
                            form_trend = 0
                        features['FormTrend'] = form_trend
                        
                        # Trainer features (8)
                        trainer_id = r['TrainerID']
                        t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                        features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                        
                        cutoff_30d = race_date - timedelta(days=30)
                        cutoff_60d = race_date - timedelta(days=60)
                        t_rec = trainer_recent.get(trainer_id, [])
                        rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                        rec_60 = [x for x in t_rec if x[0] >= cutoff_60d]
                        
                        features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                        features['TrainerWinRate60d'] = safe_div(sum(x[1] for x in rec_60), len(rec_60), features['TrainerWinRate']) if len(rec_60) >= 10 else features['TrainerWinRate']
                        features['TrainerStarts60d'] = len(rec_60)
                        features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                        
                        t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                        features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                        features['TrainerTrackRuns'] = min(t_track['runs'], 100) / 100
                        
                        dist_key = round(distance / 100) * 100
                        t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                        features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                        
                        # Track/Distance features (10)
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['CareerWinRate']) if len(dist_runs) >= 3 else features['CareerWinRate']
                        features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), 0.35) if len(dist_runs) >= 3 else 0.35
                        features['DistExperience'] = min(len(dist_runs), 30) / 30
                        features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                        
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), 0.35) if len(track_runs) >= 3 else 0.35
                        features['TrackExperience'] = min(len(track_runs), 20) / 20
                        features['TrackAvgPos'] = np.mean([t['position'] for t in track_runs]) if track_runs else features['PosAvg']
                        
                        tier_runs = [h for h in hist if h.get('tier', 0) == tier]
                        features['TierExperience'] = min(len(tier_runs), 30) / 30
                        
                        box = int(r['Box'])
                        tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                        features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                        
                        # Age features (4)
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        features['AgeMonths'] = age_months
                        features['ExperiencePerAge'] = features['CareerStarts'] / (age_months + 1)
                        features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                        features['AgePeakDist'] = abs(age_months - 30)
                        
                        # Closing/Pace features (4)
                        deltas_array = deltas if deltas else [0]
                        features['PosImprovement'] = np.mean(deltas_array)
                        features['ClosingAvg'] = np.mean(closings) if closings else 0
                        features['ClosingBest'] = min(closings) if closings else 0
                        features['DeltaBest'] = max(deltas) if deltas else 0
                        
                        # Bloodline features (6)
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                        features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                        features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                        features['DamRuns'] = min(dam_data['runs'], 200) / 200
                        features['SireRuns'] = min(sire_data['runs'], 500) / 500
                        features['BloodlineVsDog'] = features['BloodlineScore'] - features['CareerWinRate']
                        
                        # Weight features (4)
                        weight_avg = np.mean(weights) if weights else 30
                        current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                        features['Weight'] = current_weight
                        features['WeightAvg'] = weight_avg
                        features['WeightChange'] = current_weight - weight_avg
                        features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                        
                        # Box features (4)
                        features['Box'] = box
                        inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                        outside_runs = [h for h in hist if h.get('box', 4) > 4]
                        inside_rate = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['CareerWinRate']) if len(inside_runs) >= 3 else features['CareerWinRate']
                        outside_rate = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['CareerWinRate']) if len(outside_runs) >= 3 else features['CareerWinRate']
                        features['BoxPreference'] = inside_rate - outside_rate
                        
                        this_box_runs = [h for h in hist if h.get('box', 0) == box]
                        features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['CareerWinRate']) if len(this_box_runs) >= 2 else features['CareerWinRate']
                        
                        # Rest features (4)
                        days_since = (race_date - hist[-1]['date']).days
                        features['DaysSinceRace'] = days_since
                        features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                        features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                        rest_score = max(0, 1 - abs(days_since - 10) / 20)
                        features['RestScore'] = rest_score
                        
                        # Base derived features from V1 (15)
                        features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                        features['Time_x_TrainerForm'] = features['TimeBest'] * features['TrainerWinRate30d']
                        features['Beyer_x_Trainer'] = features['BeyerLag1'] * features['TrainerWinRate']
                        features['Trainer_x_Track'] = features['TrainerWinRate'] * features['TrainerTrackWinRate']
                        features['Age_x_Experience'] = (age_months / 48) * (features['CareerStarts'] / 100)
                        features['Form_x_Trainer'] = form_trend * features['TrainerWinRate']
                        features['Weight_x_Distance'] = features['WeightChange'] * (1 if distance > 500 else -1)
                        features['Bloodline_x_Age'] = features['BloodlineScore'] * (1 - features['AgePeakDist'] / 20)
                        features['Rest_x_Form'] = rest_score * form_trend
                        features['SpecialistScore'] = (features['DistWinRate'] + features['ThisBoxWinRate'] + features['TrainerTrackWinRate']) / 3
                        
                        # =====================================================
                        # SECTION 2: NEW FEATURE COMBINATIONS (200+)
                        # =====================================================
                        
                        # Time x Time combinations (20)
                        features['TimeBest_x_TimeAvg'] = features['TimeBest'] * features['TimeAvg']
                        features['TimeLag1_x_TimeLag2'] = features['TimeLag1'] * features['TimeLag2']
                        features['TimeStd_x_TimeBest'] = features['TimeStd'] * features['TimeBest']
                        features['TimeImproving_x_TimeBest'] = features['TimeImproving'] * features['TimeBest']
                        features['TimeTrend3_x_TimeAvg'] = features['TimeTrend3'] * features['TimeAvg']
                        features['TimeQ25_x_TimeAvg'] = features['TimeQ25'] * features['TimeAvg']
                        features['TimeBestRecent3_x_TimeBest'] = features['TimeBestRecent3'] * features['TimeBest']
                        features['Time_Range'] = features['TimeWorst'] - features['TimeBest']
                        features['Time_Momentum'] = features['TimeLag1'] - features['TimeLag3'] if len(times) >= 3 else 0
                        features['Time_Consistency'] = 1 / (features['TimeStd'] + 0.01)
                        
                        # Split x Time combinations (15)
                        features['SplitBest_x_TimeBest'] = features['SplitBest'] * features['TimeBest']
                        features['SplitAvg_x_TimeAvg'] = features['SplitAvg'] * features['TimeAvg']
                        features['SplitLag1_x_TimeLag1'] = features['SplitLag1'] * features['TimeLag1']
                        features['SplitStd_x_TimeStd'] = features['SplitStd'] * features['TimeStd']
                        features['Split_vs_Closing'] = features['SplitAvg'] - features['ClosingAvg']
                        features['EarlySpeed_Ratio'] = safe_div(features['SplitBest'], features['TimeBest'], 0.5)
                        features['LateSpeed_Ratio'] = safe_div(features['ClosingBest'], features['TimeBest'], 0.5)
                        
                        # Trainer combinations (30)
                        features['Trainer30d_x_Track'] = features['TrainerWinRate30d'] * features['TrainerTrackWinRate']
                        features['Trainer60d_x_Dist'] = features['TrainerWinRate60d'] * features['TrainerDistWinRate']
                        features['TrainerFormVsAll_x_Starts'] = features['TrainerFormVsAll'] * features['TrainerStarts60d']
                        features['TrainerTrack_x_Dist'] = features['TrainerTrackWinRate'] * features['TrainerDistWinRate']
                        features['Trainer_x_DogWinRate'] = features['TrainerWinRate'] * features['CareerWinRate']
                        features['Trainer30_x_DogWinRate'] = features['TrainerWinRate30d'] * features['CareerWinRate']
                        features['Trainer_x_DogPlaceRate'] = features['TrainerWinRate'] * features['DistPlaceRate']
                        features['TrainerHot'] = 1 if features['TrainerWinRate30d'] > features['TrainerWinRate'] * 1.2 else 0
                        features['TrainerCold'] = 1 if features['TrainerWinRate30d'] < features['TrainerWinRate'] * 0.8 else 0
                        features['TrainerTrackSpecialist'] = 1 if features['TrainerTrackWinRate'] > features['TrainerWinRate'] * 1.3 else 0
                        features['TrainerDistSpecialist'] = 1 if features['TrainerDistWinRate'] > features['TrainerWinRate'] * 1.3 else 0
                        features['Trainer_x_Bloodline'] = features['TrainerWinRate'] * features['BloodlineScore']
                        features['Trainer30_x_Bloodline'] = features['TrainerWinRate30d'] * features['BloodlineScore']
                        features['Trainer_ActivityRatio'] = safe_div(features['TrainerStarts60d'], 60, 0)
                        
                        # Box combinations (25)
                        features['Box_x_TrainerTrack'] = features['TrackBoxWinRate'] * features['TrainerTrackWinRate']
                        features['Box_x_TrainerDist'] = features['TrackBoxWinRate'] * features['TrainerDistWinRate']
                        features['Box_x_DogWinRate'] = features['TrackBoxWinRate'] * features['CareerWinRate']
                        features['Box_x_DistWinRate'] = features['TrackBoxWinRate'] * features['DistWinRate']
                        features['BoxPref_x_WinRate'] = features['BoxPreference'] * features['CareerWinRate']
                        features['ThisBox_x_TrainerTrack'] = features['ThisBoxWinRate'] * features['TrainerTrackWinRate']
                        features['ThisBox_x_TimeBest'] = features['ThisBoxWinRate'] * (1 - features['TimeBest'])
                        features['Box_x_EarlySpeed'] = features['TrackBoxWinRate'] * features['SplitBest']
                        features['IsBox1'] = 1 if box == 1 else 0
                        features['IsBox2'] = 1 if box == 2 else 0
                        features['IsBox8'] = 1 if box == 8 else 0
                        features['IsInsideBox'] = 1 if box <= 4 else 0
                        features['IsOutsideBox'] = 1 if box >= 7 else 0
                        features['IsMiddleBox'] = 1 if 3 <= box <= 6 else 0
                        features['Box_x_Closing'] = box * features['ClosingAvg']
                        features['Box_x_PosImprove'] = box * features['PosImprovement']
                        
                        # Age combinations (20)
                        features['Age_x_WinRate'] = age_months * features['CareerWinRate']
                        features['Age_x_TrainerWinRate'] = age_months * features['TrainerWinRate']
                        features['Age_x_TimeBest'] = age_months * features['TimeBest']
                        features['Age_x_Beyer'] = age_months * features['BeyerLag1']
                        features['AgePeak_x_WinRate'] = (1 - features['AgePeakDist'] / 20) * features['CareerWinRate']
                        features['AgePeak_x_Trainer'] = (1 - features['AgePeakDist'] / 20) * features['TrainerWinRate']
                        features['IsPeakAge'] = 1 if 24 <= age_months <= 36 else 0
                        features['IsYoung'] = 1 if age_months < 24 else 0
                        features['IsVeteran'] = 1 if age_months > 42 else 0
                        features['IsPrime'] = 1 if 28 <= age_months <= 34 else 0
                        features['Age_x_Experience'] = age_months * features['CareerStarts']
                        features['Age_x_DistExp'] = age_months * features['DistExperience']
                        features['ExpPerAge_x_WinRate'] = features['ExperiencePerAge'] * features['CareerWinRate']
                        
                        # Bloodline combinations (20)
                        features['Sire_x_Dam'] = features['SireWinRate'] * features['DamWinRate']
                        features['Sire_x_Trainer'] = features['SireWinRate'] * features['TrainerWinRate']
                        features['Dam_x_Trainer'] = features['DamWinRate'] * features['TrainerWinRate']
                        features['Bloodline_x_WinRate'] = features['BloodlineScore'] * features['CareerWinRate']
                        features['Bloodline_x_DistWinRate'] = features['BloodlineScore'] * features['DistWinRate']
                        features['Bloodline_x_TimeBest'] = features['BloodlineScore'] * (1 - features['TimeBest'])
                        features['BloodlineVsDog_x_Trainer'] = features['BloodlineVsDog'] * features['TrainerWinRate']
                        features['SireRuns_x_SireWin'] = features['SireRuns'] * features['SireWinRate']
                        features['DamRuns_x_DamWin'] = features['DamRuns'] * features['DamWinRate']
                        features['TopSire'] = 1 if features['SireWinRate'] > 0.15 else 0
                        features['TopDam'] = 1 if features['DamWinRate'] > 0.15 else 0
                        features['EliteBloodline'] = 1 if features['BloodlineScore'] > 0.14 else 0
                        
                        # Weight combinations (15)
                        features['Weight_x_TimeBest'] = features['Weight'] * features['TimeBest']
                        features['Weight_x_Beyer'] = features['Weight'] * features['BeyerLag1']
                        features['Weight_x_WinRate'] = features['Weight'] * features['CareerWinRate']
                        features['WeightChange_x_Form'] = features['WeightChange'] * form_trend
                        features['WeightChange_x_Rest'] = features['WeightChange'] * features['DaysSinceRace']
                        features['WeightStd_x_FormTrend'] = features['WeightStd'] * form_trend
                        features['IsHeavy'] = 1 if current_weight > 32 else 0
                        features['IsLight'] = 1 if current_weight < 28 else 0
                        features['WeightOptimal'] = 1 if 29 <= current_weight <= 31 else 0
                        features['WeightGain'] = 1 if features['WeightChange'] > 0.5 else 0
                        features['WeightLoss'] = 1 if features['WeightChange'] < -0.5 else 0
                        
                        # Rest/Frequency combinations (20)
                        features['Days_x_WinRate'] = features['DaysSinceRace'] * features['CareerWinRate']
                        features['Days_x_Trainer'] = features['DaysSinceRace'] * features['TrainerWinRate']
                        features['Days_x_FormTrend'] = features['DaysSinceRace'] * form_trend
                        features['Freq30_x_WinRate'] = features['RaceFrequency30d'] * features['CareerWinRate']
                        features['Freq60_x_WinRate'] = features['RaceFrequency60d'] * features['CareerWinRate']
                        features['Rest_x_Trainer'] = rest_score * features['TrainerWinRate']
                        features['Rest_x_WinRate'] = rest_score * features['CareerWinRate']
                        features['IsFresh'] = 1 if 7 <= days_since <= 14 else 0
                        features['IsBackedUp'] = 1 if days_since <= 5 else 0
                        features['IsSpelled'] = 1 if days_since > 28 else 0
                        features['IsLongSpell'] = 1 if days_since > 60 else 0
                        features['OptimalRest'] = 1 if 8 <= days_since <= 12 else 0
                        features['FrequencyRatio'] = safe_div(features['RaceFrequency30d'], features['RaceFrequency60d'], 0.5)
                        
                        # Track/Distance combinations (25)
                        features['DistWin_x_TrackWin'] = features['DistWinRate'] * features['TrackPlaceRate']
                        features['DistExp_x_TrackExp'] = features['DistExperience'] * features['TrackExperience']
                        features['DistAvg_x_TrackAvg'] = features['DistAvgPos'] * features['TrackAvgPos']
                        features['DistPlace_x_TrackPlace'] = features['DistPlaceRate'] * features['TrackPlaceRate']
                        features['TierExp_x_WinRate'] = features['TierExperience'] * features['CareerWinRate']
                        features['DistWin_x_Trainer'] = features['DistWinRate'] * features['TrainerWinRate']
                        features['TrackPlace_x_Trainer'] = features['TrackPlaceRate'] * features['TrainerWinRate']
                        features['IsDistSpecialist'] = 1 if features['DistWinRate'] > features['CareerWinRate'] * 1.3 else 0
                        features['IsTrackSpecialist'] = 1 if features['TrackPlaceRate'] > 0.45 else 0
                        features['DistTrackCombo'] = features['DistWinRate'] * features['TrackPlaceRate'] * features['TrainerTrackWinRate']
                        
                        # Closing/Pace combinations (20)
                        features['Closing_x_TrainerTrack'] = features['ClosingAvg'] * features['TrainerTrackWinRate']
                        features['Closing_x_WinRate'] = features['ClosingAvg'] * features['CareerWinRate']
                        features['Closing_x_Box'] = features['ClosingAvg'] * box
                        features['DeltaBest_x_WinRate'] = features['DeltaBest'] * features['CareerWinRate']
                        features['PosImprove_x_Trainer'] = features['PosImprovement'] * features['TrainerWinRate']
                        features['IsCloser'] = 1 if features['PosImprovement'] > 1 else 0
                        features['IsLeader'] = 1 if features['PosImprovement'] < -1 else 0
                        features['IsMidPack'] = 1 if -1 <= features['PosImprovement'] <= 1 else 0
                        features['Closer_x_WideBox'] = features['IsCloser'] * features['IsOutsideBox']
                        features['Leader_x_RailBox'] = features['IsLeader'] * features['IsInsideBox']
                        
                        # Beyer combinations (15)
                        features['Beyer_x_WinRate'] = features['BeyerLag1'] * features['CareerWinRate']
                        features['Beyer_x_DistWin'] = features['BeyerLag1'] * features['DistWinRate']
                        features['Beyer_x_TrainerTrack'] = features['BeyerLag1'] * features['TrainerTrackWinRate']
                        features['Beyer_x_Bloodline'] = features['BeyerLag1'] * features['BloodlineScore']
                        features['Beyer_x_Age'] = features['BeyerLag1'] * age_months
                        features['BeyerStd_x_WinRate'] = features['BeyerStd'] * features['CareerWinRate']
                        features['Beyer_PeakAge'] = features['BeyerLag1'] * features['IsPeakAge']
                        features['Beyer_Fresh'] = features['BeyerLag1'] * features['IsFresh']
                        
                        # =====================================================
                        # SECTION 3: RATIO FEATURES (30)
                        # =====================================================
                        features['WinRate_vs_Trainer'] = safe_div(features['CareerWinRate'], features['TrainerWinRate'], 1)
                        features['WinRate_vs_Blood'] = safe_div(features['CareerWinRate'], features['BloodlineScore'], 1)
                        features['DistWin_vs_Career'] = safe_div(features['DistWinRate'], features['CareerWinRate'], 1)
                        features['TrackWin_vs_Career'] = safe_div(features['TrackPlaceRate'], features['CareerWinRate'], 1)
                        features['Trainer30_vs_All'] = safe_div(features['TrainerWinRate30d'], features['TrainerWinRate'], 1)
                        features['Trainer60_vs_All'] = safe_div(features['TrainerWinRate60d'], features['TrainerWinRate'], 1)
                        features['TrainerTrack_vs_All'] = safe_div(features['TrainerTrackWinRate'], features['TrainerWinRate'], 1)
                        features['TrainerDist_vs_All'] = safe_div(features['TrainerDistWinRate'], features['TrainerWinRate'], 1)
                        features['ThisBox_vs_Career'] = safe_div(features['ThisBoxWinRate'], features['CareerWinRate'], 1)
                        features['TimeLag1_vs_Best'] = safe_div(features['TimeLag1'], features['TimeBest'], 1)
                        features['TimeLag1_vs_Avg'] = safe_div(features['TimeLag1'], features['TimeAvg'], 1)
                        features['SplitLag1_vs_Best'] = safe_div(features['SplitLag1'], features['SplitBest'], 1) if features['SplitBest'] != 0 else 1
                        features['ClosingBest_vs_Avg'] = safe_div(features['ClosingBest'], features['ClosingAvg'], 1) if features['ClosingAvg'] != 0 else 1
                        features['Weight_vs_Avg'] = safe_div(features['Weight'], features['WeightAvg'], 1) if features['WeightAvg'] != 0 else 1
                        features['Sire_vs_Dam'] = safe_div(features['SireWinRate'], features['DamWinRate'], 1)
                        
                        # =====================================================
                        # SECTION 4: DIFFERENCE FEATURES (20)
                        # =====================================================
                        features['WinRate_minus_Trainer'] = features['CareerWinRate'] - features['TrainerWinRate']
                        features['WinRate_minus_Blood'] = features['CareerWinRate'] - features['BloodlineScore']
                        features['DistWin_minus_Career'] = features['DistWinRate'] - features['CareerWinRate']
                        features['TrackPlace_minus_Career'] = features['TrackPlaceRate'] - 0.35
                        features['Trainer30_minus_All'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                        features['ThisBox_minus_Career'] = features['ThisBoxWinRate'] - features['CareerWinRate']
                        features['TimeLag1_minus_Best'] = features['TimeLag1'] - features['TimeBest']
                        features['TimeLag1_minus_Avg'] = features['TimeLag1'] - features['TimeAvg']
                        features['SplitLag1_minus_Best'] = features['SplitLag1'] - features['SplitBest']
                        features['Weight_minus_Avg'] = features['Weight'] - features['WeightAvg']
                        features['Sire_minus_Dam'] = features['SireWinRate'] - features['DamWinRate']
                        
                        # =====================================================
                        # SECTION 5: POLYNOMIAL FEATURES (25)
                        # =====================================================
                        features['WinRate_sq'] = features['CareerWinRate'] ** 2
                        features['TrainerWinRate_sq'] = features['TrainerWinRate'] ** 2
                        features['BloodlineScore_sq'] = features['BloodlineScore'] ** 2
                        features['TimeBest_sq'] = features['TimeBest'] ** 2
                        features['BeyerLag1_sq'] = (features['BeyerLag1'] / 100) ** 2
                        features['PosAvg_sq'] = features['PosAvg'] ** 2
                        features['DaysSinceRace_sq'] = (features['DaysSinceRace'] / 30) ** 2
                        features['Age_sq'] = (age_months / 36) ** 2
                        features['FormTrend_sq'] = form_trend ** 2
                        features['WeightChange_sq'] = features['WeightChange'] ** 2
                        features['DistWinRate_sq'] = features['DistWinRate'] ** 2
                        features['TrackBoxWinRate_sq'] = features['TrackBoxWinRate'] ** 2
                        
                        # Cubed features for key variables
                        features['WinRate_cb'] = features['CareerWinRate'] ** 3
                        features['TrainerWinRate_cb'] = features['TrainerWinRate'] ** 3
                        
                        # =====================================================
                        # SECTION 6: COMPOSITE SCORES (30)
                        # =====================================================
                        features['QualityScore1'] = (features['BeyerLag1'] / 100) * features['CareerWinRate'] * features['TrainerWinRate'] * 100
                        features['QualityScore2'] = features['TimeBest'] * features['CareerWinRate'] * features['TrainerTrackWinRate']
                        features['FormQuality1'] = features['WinRate5'] * features['TrainerWinRate30d']
                        features['FormQuality2'] = form_trend * features['TrainerWinRate30d'] * features['CareerWinRate']
                        features['SpecialistScore2'] = (features['DistWinRate'] + features['TrackPlaceRate'] + features['TrainerTrackWinRate']) / 3
                        features['SpecialistScore3'] = features['DistWinRate'] * features['TrackPlaceRate'] * features['ThisBoxWinRate']
                        features['AllRoundScore'] = (features['CareerWinRate'] + features['DistWinRate'] + features['TrainerWinRate']) / 3
                        features['ConditionScore'] = features['IsFresh'] * features['WeightOptimal'] * features['IsPeakAge']
                        features['ReadyToWin'] = features['TrainerHot'] * features['IsFresh'] * (1 if form_trend > 0 else 0)
                        features['ConfidenceScore'] = features['TrainerTrackWinRate'] * features['ThisBoxWinRate'] * features['DistWinRate']
                        features['ValueScore'] = features['CareerWinRate'] * features['BloodlineScore'] * (1 - features['AgePeakDist'] / 20)
                        features['TotalScore'] = (features['QualityScore1'] + features['SpecialistScore2'] + features['AllRoundScore']) / 3
                        
                        # Momentum/trend composites
                        features['MomentumScore'] = form_trend * (1 if features['TrainerWinRate30d'] > features['TrainerWinRate'] else 0)
                        features['RisingScore'] = features['TimeImproving'] * form_trend * -1  # Negative = better
                        features['PeakPerformance'] = features['IsPeakAge'] * features['IsFresh'] * features['TrainerHot']
                        
                        feature_rows.append(features)
        
        # Update lookups (same as before)
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            box = int(r['Box']) if pd.notna(r['Box']) else 4
            won = r['Won']
            
            dog_history[dog_id].append({
                'date': race_date, 'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
                'position': r['Position'], 'track_id': track_id, 'distance': distance, 'tier': tier,
                'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
                'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
                'weight': r['Weight'], 'box': box, 'split': r['Split'] if pd.notna(r['Split']) else None
            })
            
            if pd.notna(r['TrainerID']):
                tid = r['TrainerID']
                trainer_all[tid]['runs'] += 1
                if won: trainer_all[tid]['wins'] += 1
                trainer_recent[tid].append((race_date, won, track_id, distance))
                cutoff = race_date - timedelta(days=120)
                trainer_recent[tid] = [x for x in trainer_recent[tid] if x[0] >= cutoff]
                trainer_track[tid][track_id]['runs'] += 1
                if won: trainer_track[tid][track_id]['wins'] += 1
                dist_key = round(distance / 100) * 100
                trainer_dist[tid][dist_key]['runs'] += 1
                if won: trainer_dist[tid][dist_key]['wins'] += 1
            
            if pd.notna(r['SireID']):
                sire_stats[r['SireID']]['runs'] += 1
                if won: sire_stats[r['SireID']]['wins'] += 1
            if pd.notna(r['DamID']):
                dam_stats[r['DamID']]['runs'] += 1
                if won: dam_stats[r['DamID']]['wins'] += 1
            
            track_box_wins[track_id][box]['runs'] += 1
            if won: track_box_wins[track_id][box]['wins'] += 1
        
        processed += 1
        if processed % 50000 == 0:
            print(f"  {processed:,} races...")
    
    print(f"  Total: {processed:,} races")
    print(f"  Features: {len(feature_rows):,}")
    
    print("\n[5/7] Preparing data...")
    feat_df = pd.DataFrame(feature_rows)
    feat_df['BSP'] = pd.to_numeric(feat_df['BSP'], errors='coerce')
    feat_df = feat_df[(feat_df['TimeBest'] > -5) & (feat_df['TimeBest'] < 5)]
    
    meta_cols = ['RaceID', 'Won', 'BSP', 'Tier']
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    print(f"Total features generated: {len(feature_cols)}")
    
    train_df = feat_df[feat_df['RaceID'] % 3 != 0]
    test_df = feat_df[feat_df['RaceID'] % 3 == 0]
    
    if len(train_df) > 50000:
        print(f"  Subsampling from {len(train_df):,} to 50,000...")
        train_df = train_df.sample(n=50000, random_state=42)
    
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['Won']
    
    print("\n[6/7] Quick training for feature ranking (100 iterations)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_dist = {
        'n_estimators': randint(100, 200),
        'max_depth': randint(4, 6),
        'learning_rate': uniform(0.08, 0.12),
        'min_child_samples': randint(50, 150),
        'subsample': uniform(0.7, 0.2),
        'colsample_bytree': uniform(0.7, 0.2),
        'reg_alpha': uniform(0, 0.5),
        'reg_lambda': uniform(0, 0.5)
    }
    
    lgb = LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, random_state=42, verbose=-1)
    
    random_search = RandomizedSearchCV(lgb, param_dist, n_iter=100, cv=3, scoring='roc_auc', n_jobs=1, verbose=0, random_state=42)
    print("  Training 100 iterations...")
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest CV AUC: {random_search.best_score_:.4f}")
    
    print("\n[7/7] Ranking features by importance...")
    
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance['CumulativeImportance'] = importance['Importance'].cumsum() / importance['Importance'].sum() * 100
    importance['Rank'] = range(1, len(importance) + 1)
    
    print("\n" + "="*70)
    print("TOP 50 FEATURES (Ranked by Importance)")
    print("="*70)
    print(importance.head(50).to_string(index=False))
    
    print("\n" + "="*70)
    print("FEATURES WITH <0.5% IMPORTANCE (Candidates for removal)")
    print("="*70)
    low_importance = importance[importance['Importance'] < importance['Importance'].sum() * 0.005]
    print(f"Count: {len(low_importance)}")
    
    importance.to_csv('models/feature_importance_ranking_v2.csv', index=False)
    print(f"\nSaved ranking to models/feature_importance_ranking_v2.csv")
    
    top_features = importance[importance['CumulativeImportance'] <= 80]['Feature'].tolist()
    with open('models/top_features_v2.txt', 'w') as f:
        f.write('\n'.join(top_features))
    print(f"Saved top {len(top_features)} features (80% cumulative) to models/top_features_v2.txt")
    
    # Quick backtest
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print("\n" + "="*70)
    print("QUICK BACKTEST (Feature Factory V2 Model)")
    print("="*70)
    
    def backtest(df, label):
        if len(df) < 50: return
        wins = df['Won'].sum()
        sr = wins / len(df) * 100
        valid = df.dropna(subset=['BSP'])
        returns = valid[valid['Won'] == 1]['BSP'].sum()
        profit = returns - len(valid)
        roi = profit / len(valid) * 100 if len(valid) > 0 else 0
        print(f"{label}: {len(df):,} bets, {wins:,} wins ({sr:.1f}%), ROI: {roi:+.1f}%")
    
    backtest(race_leaders, "All picks")
    backtest(race_leaders[(race_leaders['BSP'] >= 2) & (race_leaders['BSP'] <= 10)], "$2-$10")
    backtest(race_leaders[(race_leaders['BSP'] >= 3) & (race_leaders['BSP'] <= 8)], "$3-$8")
    
    print("="*70)

if __name__ == "__main__":
    train_feature_factory_v2()
