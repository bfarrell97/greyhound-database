"""
Feature Factory - Comprehensive Feature Generation & Selection
==============================================================
Generates 150+ candidate features, trains a model, and ranks
by importance to identify the most predictive features.

Feature Categories:
1. Time/Speed features (20+)
2. Position/Form features (20+)
3. Trainer features (15+)
4. Track/Distance features (15+)
5. Age/Career features (15+)
6. Closing/Pace features (15+)
7. Bloodline features (10+)
8. Weight/Physical features (10+)
9. Box/Field features (15+)
10. Market features (if available) (10+)
11. Derived/Interaction features (30+)

Uses quick training (100 iterations) to rank features,
then outputs the top performers for use in production models.
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
    """Safe division avoiding div by zero"""
    return a / b if b != 0 else default

def train_feature_factory():
    print("="*70)
    print("FEATURE FACTORY - 150+ CANDIDATE FEATURES")
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
    
    print("\n[3/7] Building comprehensive lookups...")
    
    dog_history = defaultdict(list)
    trainer_all = defaultdict(lambda: {'wins': 0, 'runs': 0})
    trainer_recent = defaultdict(list)
    trainer_track = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    trainer_dist = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    sire_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    dam_stats = defaultdict(lambda: {'wins': 0, 'runs': 0})
    track_box_wins = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'runs': 0}))
    
    print("\n[4/7] Building 150+ features...")
    
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
                    recent_5 = hist[-5:]
                    recent_3 = hist[-3:]
                    
                    # Extract data from history
                    times = [h['norm_time'] for h in recent if h['norm_time'] is not None]
                    beyers = [h['beyer'] for h in recent if h['beyer'] is not None]
                    positions = [h['position'] for h in recent if h['position'] is not None]
                    closings = [h['closing'] for h in recent if h['closing'] is not None]
                    deltas = [h['pos_delta'] for h in recent if h['pos_delta'] is not None]
                    weights = [h['weight'] for h in recent if h['weight'] is not None and h['weight'] > 0]
                    splits = [h['split'] for h in recent if h['split'] is not None]
                    
                    if len(times) >= 3:
                        features = {'RaceID': race_id, 'Won': r['Won'], 'BSP': r['BSP'], 'Tier': tier}
                        
                        # ===== 1. TIME/SPEED FEATURES (25) =====
                        features['TimeBest'] = min(times)
                        features['TimeWorst'] = max(times)
                        features['TimeAvg'] = np.mean(times)
                        features['TimeAvg3'] = np.mean(times[-3:])
                        features['TimeAvg5'] = np.mean(times[-5:]) if len(times) >= 5 else np.mean(times)
                        features['TimeLag1'] = times[-1]
                        features['TimeLag2'] = times[-2] if len(times) >= 2 else times[-1]
                        features['TimeLag3'] = times[-3] if len(times) >= 3 else times[-1]
                        features['TimeStd'] = np.std(times) if len(times) >= 3 else 0
                        features['TimeRange'] = max(times) - min(times)
                        features['TimeImproving'] = times[-1] - times[0] if len(times) >= 2 else 0  # Negative = improving
                        features['TimeTrend3'] = (times[-1] - times[-3]) if len(times) >= 3 else 0
                        features['TimeBestRecent3'] = min(times[-3:])
                        features['TimeMedian'] = np.median(times)
                        features['TimeQ25'] = np.percentile(times, 25) if len(times) >= 4 else min(times)
                        
                        # Split/early speed
                        features['SplitBest'] = min(splits) if splits else 0
                        features['SplitAvg'] = np.mean(splits) if splits else 0
                        features['SplitLag1'] = splits[-1] if splits else 0
                        features['SplitStd'] = np.std(splits) if len(splits) >= 3 else 0
                        
                        # Beyer speed figures
                        features['BeyerBest'] = max(beyers) if beyers else 77
                        features['BeyerAvg'] = np.mean(beyers) if beyers else 77
                        features['BeyerLag1'] = beyers[-1] if beyers else 77
                        features['BeyerLag2'] = beyers[-2] if len(beyers) >= 2 else features['BeyerLag1']
                        features['BeyerStd'] = np.std(beyers) if len(beyers) >= 3 else 0
                        features['BeyerTrend'] = (beyers[-1] - beyers[0]) if len(beyers) >= 2 else 0
                        
                        # ===== 2. POSITION/FORM FEATURES (25) =====
                        features['PosAvg'] = np.mean(positions)
                        features['PosAvg3'] = np.mean(positions[-3:])
                        features['PosAvg5'] = np.mean(positions[-5:]) if len(positions) >= 5 else features['PosAvg']
                        features['PosLag1'] = positions[-1]
                        features['PosLag2'] = positions[-2] if len(positions) >= 2 else positions[-1]
                        features['PosBest'] = min(positions)
                        features['PosWorst'] = max(positions)
                        features['PosStd'] = np.std(positions) if len(positions) >= 3 else 0
                        features['PosMedian'] = np.median(positions)
                        
                        # Win/place rates
                        features['WinRate'] = sum(1 for p in positions if p == 1) / len(positions)
                        features['WinRate3'] = sum(1 for p in positions[-3:] if p == 1) / 3
                        features['WinRate5'] = sum(1 for p in positions[-5:] if p == 1) / min(5, len(positions))
                        features['PlaceRate'] = sum(1 for p in positions if p <= 3) / len(positions)
                        features['PlaceRate3'] = sum(1 for p in positions[-3:] if p <= 3) / 3
                        features['Top4Rate'] = sum(1 for p in positions if p <= 4) / len(positions)
                        
                        # Streaks and consistency
                        features['WinStreak'] = 0
                        for p in reversed(positions):
                            if p == 1: features['WinStreak'] += 1
                            else: break
                        features['PlaceStreak'] = 0
                        for p in reversed(positions):
                            if p <= 3: features['PlaceStreak'] += 1
                            else: break
                        features['LossStreak'] = 0
                        for p in reversed(positions):
                            if p > 3: features['LossStreak'] += 1
                            else: break
                        
                        # Form trend
                        if len(positions) >= 5:
                            first_half = np.mean(positions[:len(positions)//2])
                            second_half = np.mean(positions[len(positions)//2:])
                            features['FormTrend'] = first_half - second_half  # Positive = improving
                        else:
                            features['FormTrend'] = 0
                        
                        features['ConsistencyScore'] = 1 / (features['PosStd'] + 0.1)
                        features['LastWonDaysAgo'] = 999
                        for i, h in enumerate(reversed(hist)):
                            if h['position'] == 1:
                                features['LastWonDaysAgo'] = (race_date - h['date']).days
                                break
                        
                        # ===== 3. TRAINER FEATURES (15) =====
                        trainer_id = r['TrainerID']
                        t_all = trainer_all.get(trainer_id, {'wins': 0, 'runs': 0})
                        features['TrainerWinRate'] = safe_div(t_all['wins'], t_all['runs'], 0.12)
                        features['TrainerRuns'] = min(t_all['runs'], 500) / 500
                        
                        cutoff_30d = race_date - timedelta(days=30)
                        cutoff_60d = race_date - timedelta(days=60)
                        cutoff_90d = race_date - timedelta(days=90)
                        t_rec = trainer_recent.get(trainer_id, [])
                        rec_30 = [x for x in t_rec if x[0] >= cutoff_30d]
                        rec_60 = [x for x in t_rec if x[0] >= cutoff_60d]
                        rec_90 = [x for x in t_rec if x[0] >= cutoff_90d]
                        
                        features['TrainerWinRate30d'] = safe_div(sum(x[1] for x in rec_30), len(rec_30), features['TrainerWinRate']) if len(rec_30) >= 5 else features['TrainerWinRate']
                        features['TrainerWinRate60d'] = safe_div(sum(x[1] for x in rec_60), len(rec_60), features['TrainerWinRate']) if len(rec_60) >= 10 else features['TrainerWinRate']
                        features['TrainerWinRate90d'] = safe_div(sum(x[1] for x in rec_90), len(rec_90), features['TrainerWinRate']) if len(rec_90) >= 15 else features['TrainerWinRate']
                        features['TrainerStarts30d'] = len(rec_30)
                        features['TrainerStarts60d'] = len(rec_60)
                        
                        t_track = trainer_track.get(trainer_id, {}).get(track_id, {'wins': 0, 'runs': 0})
                        features['TrainerTrackWinRate'] = safe_div(t_track['wins'], t_track['runs'], features['TrainerWinRate']) if t_track['runs'] >= 10 else features['TrainerWinRate']
                        features['TrainerTrackRuns'] = min(t_track['runs'], 100) / 100
                        
                        dist_key = round(distance / 100) * 100
                        t_dist = trainer_dist.get(trainer_id, {}).get(dist_key, {'wins': 0, 'runs': 0})
                        features['TrainerDistWinRate'] = safe_div(t_dist['wins'], t_dist['runs'], features['TrainerWinRate']) if t_dist['runs'] >= 10 else features['TrainerWinRate']
                        
                        features['TrainerFormVsAll'] = features['TrainerWinRate30d'] - features['TrainerWinRate']
                        features['TrainerHot'] = 1 if features['TrainerWinRate30d'] > features['TrainerWinRate'] * 1.2 else 0
                        features['TrainerCold'] = 1 if features['TrainerWinRate30d'] < features['TrainerWinRate'] * 0.8 else 0
                        
                        # ===== 4. TRACK/DISTANCE FEATURES (15) =====
                        dist_runs = [h for h in hist if abs(h['distance'] - distance) < 50]
                        features['DistWinRate'] = safe_div(sum(1 for d in dist_runs if d['position'] == 1), len(dist_runs), features['WinRate']) if len(dist_runs) >= 3 else features['WinRate']
                        features['DistPlaceRate'] = safe_div(sum(1 for d in dist_runs if d['position'] <= 3), len(dist_runs), features['PlaceRate']) if len(dist_runs) >= 3 else features['PlaceRate']
                        features['DistExperience'] = min(len(dist_runs), 30) / 30
                        features['DistAvgPos'] = np.mean([d['position'] for d in dist_runs]) if dist_runs else features['PosAvg']
                        
                        track_runs = [h for h in hist if h['track_id'] == track_id]
                        features['TrackWinRate'] = safe_div(sum(1 for t in track_runs if t['position'] == 1), len(track_runs), features['WinRate']) if len(track_runs) >= 3 else features['WinRate']
                        features['TrackPlaceRate'] = safe_div(sum(1 for t in track_runs if t['position'] <= 3), len(track_runs), features['PlaceRate']) if len(track_runs) >= 3 else features['PlaceRate']
                        features['TrackExperience'] = min(len(track_runs), 20) / 20
                        features['TrackAvgPos'] = np.mean([t['position'] for t in track_runs]) if track_runs else features['PosAvg']
                        
                        # Track specialists
                        features['IsDistSpecialist'] = 1 if features['DistWinRate'] > features['WinRate'] * 1.3 else 0
                        features['IsTrackSpecialist'] = 1 if features['TrackWinRate'] > features['WinRate'] * 1.3 else 0
                        features['TrackDistCombo'] = features['TrackWinRate'] * features['DistWinRate']
                        
                        # Tier experience
                        tier_runs = [h for h in hist if h.get('tier', 0) == tier]
                        features['TierWinRate'] = safe_div(sum(1 for t in tier_runs if t['position'] == 1), len(tier_runs), features['WinRate']) if len(tier_runs) >= 5 else features['WinRate']
                        features['TierExperience'] = min(len(tier_runs), 30) / 30
                        
                        # Track box bias
                        box = int(r['Box'])
                        tb = track_box_wins.get(track_id, {}).get(box, {'wins': 0, 'runs': 0})
                        features['TrackBoxWinRate'] = safe_div(tb['wins'], tb['runs'], 0.125) if tb['runs'] >= 50 else 0.125
                        
                        # ===== 5. AGE/CAREER FEATURES (15) =====
                        age_months = r['AgeMonths'] if pd.notna(r['AgeMonths']) else 30
                        features['AgeMonths'] = age_months
                        features['AgeYears'] = age_months / 12
                        features['IsPeakAge'] = 1 if 24 <= age_months <= 36 else 0
                        features['IsYoung'] = 1 if age_months < 24 else 0
                        features['IsVeteran'] = 1 if age_months > 42 else 0
                        features['IsPrime'] = 1 if 28 <= age_months <= 34 else 0
                        features['AgePeakDist'] = abs(age_months - 30)  # Distance from peak (30 months)
                        
                        career_starts = min(len(hist), 100)
                        features['CareerStarts'] = career_starts
                        features['CareerWins'] = sum(1 for h in hist if h['position'] == 1)
                        features['CareerWinRate'] = safe_div(features['CareerWins'], career_starts, 0.12)
                        features['CareerPlaces'] = sum(1 for h in hist if h['position'] <= 3)
                        features['CareerPlaceRate'] = safe_div(features['CareerPlaces'], career_starts, 0.35)
                        
                        features['ExperiencePerAge'] = career_starts / (age_months + 1)
                        features['WinsPerAge'] = features['CareerWins'] / (age_months + 1)
                        features['IsExperienced'] = 1 if career_starts >= 30 else 0
                        
                        # ===== 6. CLOSING/PACE FEATURES (15) =====
                        features['ClosingAvg'] = np.mean(closings) if closings else 0
                        features['ClosingBest'] = min(closings) if closings else 0
                        features['ClosingLag1'] = closings[-1] if closings else 0
                        features['ClosingStd'] = np.std(closings) if len(closings) >= 3 else 0
                        
                        features['DeltaAvg'] = np.mean(deltas) if deltas else 0
                        features['DeltaLag1'] = deltas[-1] if deltas else 0
                        features['DeltaBest'] = max(deltas) if deltas else 0  # Most positions gained
                        features['DeltaWorst'] = min(deltas) if deltas else 0  # Most positions lost
                        
                        features['IsCloser'] = 1 if features['DeltaAvg'] > 1 else 0
                        features['IsLeader'] = 1 if features['DeltaAvg'] < -1 else 0
                        features['IsMidPack'] = 1 if -1 <= features['DeltaAvg'] <= 1 else 0
                        
                        # Early vs late speed
                        if splits and closings:
                            features['EarlyVsLate'] = np.mean(splits) - np.mean(closings)
                        else:
                            features['EarlyVsLate'] = 0
                        
                        features['FirstSplitPosAvg'] = np.mean([h.get('first_split_pos', 4) for h in recent if h.get('first_split_pos')]) if any(h.get('first_split_pos') for h in recent) else 4
                        features['PosImprovement'] = features['FirstSplitPosAvg'] - features['PosAvg']
                        
                        # ===== 7. BLOODLINE FEATURES (10) =====
                        sire_data = sire_stats.get(r['SireID'], {'wins': 0, 'runs': 0})
                        dam_data = dam_stats.get(r['DamID'], {'wins': 0, 'runs': 0})
                        features['SireWinRate'] = safe_div(sire_data['wins'], sire_data['runs'], 0.12) if sire_data['runs'] > 50 else 0.12
                        features['DamWinRate'] = safe_div(dam_data['wins'], dam_data['runs'], 0.12) if dam_data['runs'] > 30 else 0.12
                        features['BloodlineScore'] = (features['SireWinRate'] + features['DamWinRate']) / 2
                        features['SireRuns'] = min(sire_data['runs'], 500) / 500
                        features['DamRuns'] = min(dam_data['runs'], 200) / 200
                        features['BloodlineQuality'] = (features['SireWinRate'] * features['SireRuns'] + features['DamWinRate'] * features['DamRuns']) / 2
                        features['TopSire'] = 1 if features['SireWinRate'] > 0.15 else 0
                        features['TopDam'] = 1 if features['DamWinRate'] > 0.15 else 0
                        features['EliteBloodline'] = 1 if features['BloodlineScore'] > 0.14 else 0
                        features['BloodlineVsDog'] = features['BloodlineScore'] - features['WinRate']
                        
                        # ===== 8. WEIGHT/PHYSICAL FEATURES (10) =====
                        weight_avg = np.mean(weights) if weights else 30
                        current_weight = r['Weight'] if pd.notna(r['Weight']) and r['Weight'] > 0 else weight_avg
                        features['Weight'] = current_weight
                        features['WeightAvg'] = weight_avg
                        features['WeightChange'] = current_weight - weight_avg
                        features['WeightStd'] = np.std(weights) if len(weights) >= 3 else 0
                        features['IsHeavy'] = 1 if current_weight > 32 else 0
                        features['IsLight'] = 1 if current_weight < 28 else 0
                        features['WeightGain'] = 1 if features['WeightChange'] > 0.5 else 0
                        features['WeightLoss'] = 1 if features['WeightChange'] < -0.5 else 0
                        features['WeightOptimal'] = 1 if 29 <= current_weight <= 31 else 0
                        features['WeightVsAvg'] = current_weight - 30  # vs population average
                        
                        # ===== 9. BOX/FIELD FEATURES (15) =====
                        features['Box'] = box
                        features['BoxType'] = 1 if box <= 4 else 0  # Inside
                        features['IsBox1'] = 1 if box == 1 else 0
                        features['IsBox8'] = 1 if box == 8 else 0
                        features['IsWideBox'] = 1 if box >= 7 else 0
                        features['IsRailBox'] = 1 if box <= 2 else 0
                        features['IsMiddleBox'] = 1 if 3 <= box <= 6 else 0
                        
                        # Box win rates from history
                        inside_runs = [h for h in hist if h.get('box', 4) <= 4]
                        outside_runs = [h for h in hist if h.get('box', 4) > 4]
                        features['InsideWinRate'] = safe_div(sum(1 for h in inside_runs if h['position'] == 1), len(inside_runs), features['WinRate']) if len(inside_runs) >= 3 else features['WinRate']
                        features['OutsideWinRate'] = safe_div(sum(1 for h in outside_runs if h['position'] == 1), len(outside_runs), features['WinRate']) if len(outside_runs) >= 3 else features['WinRate']
                        features['BoxPreference'] = features['InsideWinRate'] - features['OutsideWinRate']
                        features['BoxMatchesPref'] = 1 if (box <= 4 and features['BoxPreference'] > 0) or (box > 4 and features['BoxPreference'] < 0) else 0
                        
                        # This box history
                        this_box_runs = [h for h in hist if h.get('box', 0) == box]
                        features['ThisBoxWinRate'] = safe_div(sum(1 for h in this_box_runs if h['position'] == 1), len(this_box_runs), features['WinRate']) if len(this_box_runs) >= 2 else features['WinRate']
                        features['ThisBoxExperience'] = min(len(this_box_runs), 10) / 10
                        
                        # ===== 10. REST/FRESHNESS FEATURES (10) =====
                        days_since = (race_date - hist[-1]['date']).days
                        features['DaysSinceRace'] = days_since
                        features['IsFresh'] = 1 if 7 <= days_since <= 14 else 0
                        features['IsBackedUp'] = 1 if days_since <= 5 else 0
                        features['IsSpelled'] = 1 if days_since > 28 else 0
                        features['IsLongSpell'] = 1 if days_since > 60 else 0
                        features['OptimalRest'] = 1 if 8 <= days_since <= 12 else 0
                        features['RestScore'] = max(0, 1 - abs(days_since - 10) / 20)  # Peaks at 10 days
                        
                        # Performance after rest
                        short_rest_runs = []
                        for i in range(1, len(hist)):
                            if (hist[i]['date'] - hist[i-1]['date']).days <= 7:
                                short_rest_runs.append(hist[i])
                        if len(short_rest_runs) >= 2:
                            features['ShortRestWinRate'] = safe_div(sum(1 for h in short_rest_runs if h['position'] == 1), len(short_rest_runs), features['WinRate'])
                        else:
                            features['ShortRestWinRate'] = features['WinRate']
                        
                        features['RaceFrequency30d'] = sum(1 for h in hist if (race_date - h['date']).days <= 30)
                        features['RaceFrequency60d'] = sum(1 for h in hist if (race_date - h['date']).days <= 60)
                        
                        # ===== 11. DERIVED/INTERACTION FEATURES (30+) =====
                        # Time x Trainer
                        features['Time_x_Trainer'] = features['TimeBest'] * features['TrainerWinRate']
                        features['Time_x_TrainerForm'] = features['TimeBest'] * features['TrainerWinRate30d']
                        features['Beyer_x_Trainer'] = features['BeyerBest'] * features['TrainerWinRate']
                        
                        # Win rate combos
                        features['WinRate_x_Dist'] = features['WinRate'] * features['DistWinRate']
                        features['WinRate_x_Track'] = features['WinRate'] * features['TrackWinRate']
                        features['WinRate_x_Box'] = features['WinRate'] * features['ThisBoxWinRate']
                        features['Trainer_x_Track'] = features['TrainerWinRate'] * features['TrainerTrackWinRate']
                        
                        # Age combos
                        features['Age_x_Experience'] = (age_months / 48) * (career_starts / 100)
                        features['Age_x_Beyer'] = (age_months / 48) * (features['BeyerAvg'] / 100)
                        features['Peak_x_Form'] = features['IsPeakAge'] * features['WinRate']
                        features['Peak_x_Beyer'] = features['IsPeakAge'] * features['BeyerBest']
                        
                        # Pace combos
                        features['Closing_x_Trainer'] = features['ClosingAvg'] * features['TrainerWinRate']
                        features['Closer_x_WideBox'] = features['IsCloser'] * features['IsWideBox']
                        features['Leader_x_RailBox'] = features['IsLeader'] * features['IsRailBox']
                        features['Delta_x_Box'] = features['DeltaAvg'] * (1 if box > 4 else -1)
                        
                        # Form combos
                        features['Form_x_Trainer'] = features['FormTrend'] * features['TrainerWinRate']
                        features['Consistency_x_WinRate'] = features['ConsistencyScore'] * features['WinRate']
                        features['Streak_x_Trainer'] = features['WinStreak'] * features['TrainerWinRate']
                        
                        # Specialist combos
                        features['Specialist_x_Trainer'] = (features['IsDistSpecialist'] + features['IsTrackSpecialist']) * features['TrainerWinRate']
                        features['Track_x_Dist_Specialist'] = features['TrackWinRate'] * features['DistWinRate'] * features['TrainerTrackWinRate']
                        
                        # Weight combos
                        features['Weight_x_Distance'] = features['WeightChange'] * (1 if distance > 500 else -1)
                        features['Optimal_x_Peak'] = features['WeightOptimal'] * features['IsPeakAge']
                        
                        # Rest combos
                        features['Rest_x_Form'] = features['RestScore'] * features['FormTrend']
                        features['Fresh_x_Trainer'] = features['IsFresh'] * features['TrainerWinRate30d']
                        
                        # Bloodline combos
                        features['Bloodline_x_Age'] = features['BloodlineScore'] * (1 - features['AgePeakDist'] / 20)
                        features['Elite_x_Peak'] = features['EliteBloodline'] * features['IsPeakAge']
                        
                        # Complex combos
                        features['QualityScore'] = (features['BeyerBest'] / 100) * features['WinRate'] * features['TrainerWinRate'] * 100
                        features['FormQuality'] = features['WinRate3'] * features['TrainerWinRate30d'] * (1 + features['FormTrend'])
                        features['SpecialistScore'] = (features['DistWinRate'] + features['TrackWinRate'] + features['ThisBoxWinRate']) / 3
                        
                        feature_rows.append(features)
        
        # Update all lookups
        for _, r in race_df.iterrows():
            dog_id = r['GreyhoundID']
            box = int(r['Box']) if pd.notna(r['Box']) else 4
            won = r['Won']
            
            dog_history[dog_id].append({
                'date': race_date,
                'norm_time': r['NormTime'] if pd.notna(r['NormTime']) else None,
                'beyer': r['BeyerSpeedFigure'] if pd.notna(r['BeyerSpeedFigure']) else None,
                'position': r['Position'],
                'track_id': track_id,
                'distance': distance,
                'tier': tier,
                'closing': r['ClosingTime'] if pd.notna(r['ClosingTime']) else None,
                'pos_delta': r['PositionDelta'] if pd.notna(r['PositionDelta']) else None,
                'weight': r['Weight'],
                'box': box,
                'split': r['Split'] if pd.notna(r['Split']) else None,
                'first_split_pos': r['FirstSplitPosition'] if pd.notna(r['FirstSplitPosition']) else None
            })
            
            # Trainer updates
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
            
            # Bloodline updates
            if pd.notna(r['SireID']):
                sire_stats[r['SireID']]['runs'] += 1
                if won: sire_stats[r['SireID']]['wins'] += 1
            if pd.notna(r['DamID']):
                dam_stats[r['DamID']]['runs'] += 1
                if won: dam_stats[r['DamID']]['wins'] += 1
            
            # Track box bias
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
    
    # Get all feature columns (exclude metadata)
    meta_cols = ['RaceID', 'Won', 'BSP', 'Tier']
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    
    print(f"Total features generated: {len(feature_cols)}")
    
    train_df = feat_df[feat_df['RaceID'] % 3 != 0]
    test_df = feat_df[feat_df['RaceID'] % 3 == 0]
    
    if len(train_df) > 300000:
        print(f"  Subsampling from {len(train_df):,} to 300,000...")
        train_df = train_df.sample(n=300000, random_state=42)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Test: {len(test_df):,} samples")
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['Won']
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df['Won']
    
    print("\n[6/7] Quick training for feature ranking (100 iterations)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Quick search to rank features
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
    
    lgb = LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=42,
        verbose=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb, param_dist, n_iter=100, cv=3,
        scoring='roc_auc', n_jobs=1, verbose=1, random_state=42
    )
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest CV AUC: {random_search.best_score_:.4f}")
    
    print("\n[7/7] Ranking features by importance...")
    
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Create importance ranking
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
    print("FEATURES WITH <1% IMPORTANCE (Candidates for removal)")
    print("="*70)
    low_importance = importance[importance['Importance'] < importance['Importance'].sum() * 0.01]
    print(f"Count: {len(low_importance)}")
    print(low_importance['Feature'].tolist())
    
    # Save results
    print("\n" + "="*70)
    importance.to_csv('models/feature_importance_ranking.csv', index=False)
    print("Saved ranking to models/feature_importance_ranking.csv")
    
    # Save top features list
    top_features = importance[importance['CumulativeImportance'] <= 80]['Feature'].tolist()
    with open('models/top_features.txt', 'w') as f:
        f.write('\n'.join(top_features))
    print(f"Saved top {len(top_features)} features (80% cumulative) to models/top_features.txt")
    
    # Quick backtest
    test_df = test_df.copy()
    test_df['PredProb'] = y_pred_proba
    race_leaders = test_df.loc[test_df.groupby('RaceID')['PredProb'].idxmax()]
    
    print("\n" + "="*70)
    print("QUICK BACKTEST (Feature Factory Model)")
    print("="*70)
    
    def backtest(df, label):
        if len(df) < 50:
            return
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
    train_feature_factory()
