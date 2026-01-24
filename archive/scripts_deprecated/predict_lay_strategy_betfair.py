"""
Predict Lay Strategy Candidates (Betfair API Version)
Exact Implementation of optimize_lay_strategy.py:
1. Regression Model (predicts Time vs Benchmark)
2. Rank Predicted Times
3. Lay if Margin (1st vs 2nd) > 0.1s AND Odds < 2.50
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.betfair_fetcher import BetfairOddsFetcher
# from src.models.ml_model import GreyhoundMLModel # Removed
# from src.core.database import GreyhoundDatabase # Removed

def run_daily_predictions():
    print("="*80)
    print(f"RUNNING LAY STRATEGY PREDICTIONS: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 1. Load Model & Benchmarks
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'lay_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        model = artifacts['model']
        benchmarks = artifacts['benchmarks'] # dict: (Track, Dist) -> MedianTime
        print(f"Loaded regression model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # 2. Connect to DB and Betfair
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Betfair login failed.")
        return []
        
    db_path = os.path.join(os.path.dirname(__file__), '..', 'greyhound_racing.db')
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    # 3. Fetch Markets
    print("\n[1/3] Fetching Betfair markets...")
    now = datetime.now()
    markets = fetcher.get_greyhound_markets(from_time=now, to_time=now + timedelta(hours=24))
    
    if not markets:
        print("No markets found.")
        return []
    
    print(f"Found {len(markets)} markets.")
    
    # 4. Process Each Market
    print("\n[2/3] Analyzing markets...")
    candidates = []
    
    total_markets = len(markets)
    for i, m in enumerate(markets, 1):
        # Progress Log
        print(f"Processing Market {i}/{total_markets}: {m.market_name}")
        
        try:
            # Parse Market
            event_name = m.event.name.split(' (')[0].strip() # Track
            import re
            dist_match = re.search(r'(\d+)m', m.market_name)
            distance = int(dist_match.group(1)) if dist_match else 0
            
            race_num_match = re.search(r'R(\d+)', m.market_name)
            race_num = int(race_num_match.group(1)) if race_num_match else 0
            
            if distance == 0: continue

            # Get Benchmarks for this Track/Dist
            # Try exact match first
            median_time = benchmarks.get((event_name, distance))
            if not median_time:
                # Try finding track by partial match in benchmarks keys
                # This is tricky because benchmarks keys are (Track, Dist)
                # We assume exact match for now or skip
                # print(f"  No benchmark for {event_name} {distance}m")
                continue
                
            # Get Odds
            odds_map = fetcher.get_market_odds(m.market_id)
            
            # Prepare Race Data Frame
            race_runners = []
            
            for runner in m.runners:
                dog_name = runner.runner_name
                if " " in dog_name and dog_name[0].isdigit():
                     parts = dog_name.split(" ", 1)
                     if len(parts) > 1: dog_name = parts[1].strip()
                     
                box = None
                if hasattr(runner, 'metadata') and runner.metadata: box = runner.metadata.get('TRAP')
                if not box:
                     match = re.match(r'^(\d+)\.', runner.runner_name)
                     if match: box = match.group(1)
                box = int(box) if box else 0
                
                # Extract Feature: DogNormTimeAvg
                # Get last 3 races for this dog
                cursor = conn.cursor()
                cursor.execute("SELECT GreyhoundID FROM Greyhounds WHERE GreyhoundName = ?", (dog_name,))
                res = cursor.fetchone()
                if not res:
                     # Try case insensitive
                     cursor.execute("SELECT GreyhoundID FROM Greyhounds WHERE lower(GreyhoundName) = ?", (dog_name.lower(),))
                     res = cursor.fetchone()
                
                if not res: 
                    # print(f"    [DEBUG] Dog not found in DB: {dog_name}")
                    continue
                gid = res[0]
                
                cursor.execute("""
                    SELECT t.TrackName, r.Distance, ge.FinishTime 
                    FROM GreyhoundEntries ge
                    JOIN Races r ON ge.RaceID = r.RaceID
                    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                    JOIN Tracks t ON rm.TrackID = t.TrackID
                    WHERE ge.GreyhoundID = ? 
                      AND rm.MeetingDate < date('now')
                      AND ge.FinishTime IS NOT NULL
                    ORDER BY rm.MeetingDate DESC
                    LIMIT 3
                """, (gid,))
                
                history = cursor.fetchall()
                if len(history) < 1: 
                    # print(f"    [DEBUG] No history for {dog_name}")
                    continue 
                
                norm_times = []
                for h_track, h_dist, h_time in history:
                    h_bench = benchmarks.get((h_track, h_dist))
                    if h_bench:
                        norm_times.append(h_time - h_bench)
                    # else:
                        # print(f"    [DEBUG] No benchmark for {h_track} {h_dist}m")
                        
                if not norm_times: 
                    print(f"    [DEBUG] {dog_name}: History found but no matching benchmarks (Tracks: {[h[0] for h in history]})")
                    continue
                dog_norm_time_avg = sum(norm_times) / len(norm_times)
                
                race_runners.append({
                    'DogName': dog_name,
                    'SelectionID': runner.selection_id,
                    'Box': box,
                    'Distance': distance,
                    'DogNormTimeAvg': dog_norm_time_avg,
                    'Odds': odds_map.get(runner.selection_id, 0)
                })
                
            if not race_runners:
                print(f"  [DEBUG] No runners with sufficient history in this race.")
                continue
            
            # Predict
            df_race = pd.DataFrame(race_runners)
            features = df_race[['DogNormTimeAvg', 'Box', 'Distance']]
            
            # Predict NormTime (Lower is better)
            df_race['PredOverall'] = model.predict(features)
            
            # Rank
            df_race['PredRank'] = df_race['PredOverall'].rank(method='min')
            df_race = df_race.sort_values('PredRank')
            
            # Check Margin Logic
            if len(df_race) < 2:
                print(f"  [DEBUG] Less than 2 runners predicted.")
                continue
            
            first = df_race.iloc[0]
            second = df_race.iloc[1]
            
            margin = second['PredOverall'] - first['PredOverall']
            

            if margin > 0.1:
                # This is a candidate
                dog = first
                if dog['Odds'] > 1.0 and dog['Odds'] < 2.50:
                    candidates.append({
                         'Dog': dog['DogName'],
                         'Race': race_num,
                         'Track': event_name,
                         'Strategy': 'Lay Leading Model', 
                         'Odds': dog['Odds'],
                         'ModelProb': 0.0, 
                         'MarketID': m.market_id,
                         'SelectionID': dog['SelectionID'],
                         'StartTime': m.market_start_time,
                         'Margin': margin
                    })

        except Exception as e:
            print(f"Error processing market: {e}")
            continue

    conn.close()
    fetcher.logout()
    
    candidates.sort(key=lambda x: x['StartTime'])
    print(f"\nGenerated {len(candidates)} candidates.")
    return candidates

if __name__ == "__main__":
    results = run_daily_predictions()
    if results:
        for res in results:
            print(f"{res['Track']} R{res['Race']} - {res['Dog']} @ ${res['Odds']:.2f} [Margin {res['Margin']:.2f}] ({res['Strategy']})")
