
"""
Predict BACK Strategy Candidates (Hybrid V28/V30 + Eighth Kelly)
Uses direct Betfair Market Scanning to discover races and runners.
Populates DB with 'Race Day' data, then runs Hybrid Model (V28+V30).
Strategy: Value Threshold > 0.75, Price Cap $8.00.
Staking: Eighth Kelly ($200 Bank).
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import itertools
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.betfair_fetcher import BetfairOddsFetcher
from src.core.database import GreyhoundDatabase
from src.models.ml_model import GreyhoundMLModel

def run_daily_predictions():
    """
    Run daily predictions pipeline:
    1. Scan Betfair for upcoming races (Next 36h)
    2. Import runners/races to DB
    3. Run Hybrid ML Model (V28 + V30)
    4. Filter for Value Bets and Calculate Stakes
    """
    print("="*80)
    print(f"RUNNING DAILY PREDICTIONS (HYBRID V28/V30): {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    # Initialize
    db = GreyhoundDatabase()
    fetcher = BetfairOddsFetcher()
    
    if not fetcher.login():
        print("Fatal: Could not login to Betfair.")
        return []

    # 1. Get Markets
    print("\n[1/4] Scanning Betfair Markets...")
    markets = fetcher.get_greyhound_markets(from_time=None, to_time=None)
    
    if not markets:
        print("No markets found.")
        fetcher.logout()
        return []
        
    print(f"Found {len(markets)} markets.")
    
    # 2. Process & Import to DB
    print("\n[2/4] Importing Race Data to Database...")
    imported_count = 0
    
    for market in markets:
        try:
            event_name = market.event.name
            market_name = market.market_name
            market_start_time = market.market_start_time
            
            track_match = re.search(r'^([^(]+)', event_name)
            track_name = track_match.group(1).strip() if track_match else event_name
            race_date_str = market_start_time.strftime('%Y-%m-%d')
            
            race_num_match = re.search(r'R(\d+)', market_name)
            if not race_num_match:
                race_num_match = re.search(r'Race\s+(\d+)', market_name)
            race_number = int(race_num_match.group(1)) if race_num_match else 0
            
            if race_number == 0: continue
                
            dist_match = re.search(r'(\d+)m', market_name)
            distance = int(dist_match.group(1)) if dist_match else 0
            
            grade = market_name.replace(f"R{race_number}", "").replace(f"{distance}m", "").strip()
            market_odds = fetcher.get_market_odds(market.market_id)
            
            entries = []
            for runner in market.runners:
                runner_name = runner.runner_name
                clean_name = re.sub(r'^\d+\.\s*', '', runner_name)
                
                box = None
                if runner.metadata and 'TRAP' in runner.metadata:
                    box = runner.metadata['TRAP']
                if not box:
                    prefix_match = re.match(r'^(\d+)\.', runner_name)
                    if prefix_match: box = prefix_match.group(1)

                trainer = ""
                if runner.metadata and 'TRAINER_NAME' in runner.metadata:
                    trainer = runner.metadata['TRAINER_NAME']

                price = market_odds.get(runner.selection_id)
                    
                entries.append({
                    'greyhound_name': clean_name,
                    'box': int(box) if box else None,
                    'trainer': trainer,
                    'owner': '', 'sire': '', 'dam': '', 'starts': 0, 'wins': 0, 'prizemoney': 0,
                    'starting_price': price
                })
            
            market_time = market.market_start_time
            race_time_str = ""
            if market_time:
                try:
                    if isinstance(market_time, str):
                        dt = datetime.fromisoformat(market_time.replace('Z', '+00:00'))
                    else:
                        dt = market_time
                        if dt.tzinfo is None:
                            from datetime import timezone
                            dt = dt.replace(tzinfo=timezone.utc)
                    local_dt = dt.astimezone()
                    race_time_str = local_dt.strftime('%H:%M')
                except:
                    race_time_str = str(market_time)

            race_data = {
                'track_name': track_name, 'date': race_date_str, 'race_number': race_number,
                'race_time': race_time_str, 'race_name': market_name, 'grade': grade,
                'distance': distance, 'prize_money': '', 'entries': entries
            }
            
            if db.import_form_guide_data(race_data, race_date_str, track_name):
                imported_count += 1
                
        except Exception:
            continue

    print(f"Imported {imported_count} races into DB.")
    fetcher.logout()
    
    # 3. Predict using Autogluon V28/V30
    print("\n[3/4] Running Hybrid Model (V28/V30)...")
    
    # Load Models (Try new 'full' versions, fallback to old ones)
    try:
        predictor_v28 = TabularPredictor.load('models/autogluon_v28_full', require_version_match=False, require_py_version_match=False)
        print("[OK] Loaded V28 Full model")
    except:
        print("[WARN] V28 Full not found, using tutorial model")
        predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial', require_version_match=False, require_py_version_match=False)
        
    try:
        predictor_v30 = TabularPredictor.load('models/autogluon_v30_full', require_version_match=False, require_py_version_match=False)
        print("[OK] Loaded V30 Full model")
    except:
        print("[WARN] V30 Full not found, using BSP model")
        predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp', require_version_match=False, require_py_version_match=False)
        
    # Get today's data from DB
    today_str = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3_connect('greyhound_racing.db') # Helper below
    
    # Load Raw Data
    query = """
    SELECT
        ge.EntryID, ge.RaceID as FastTrack_RaceId, g.GreyhoundID as FastTrack_DogId,
        g.GreyhoundName, t.TrackName as Track, r.RaceNumber, r.RaceTime,
        ge.Box, ge.Weight, ge.Position as Place, ge.BSP as StartPrice, ge.PrizeMoney as Prizemoney,
        ge.FinishTime as RunTime, ge.Split as SplitMargin, ge.StartingPrice,
        r.Distance, rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = ?
    """
    df = pd.read_sql_query(query, conn, params=(today_str,))
    
    if len(df) == 0:
        print(f"No entries found for {today_str}")
        conn.close()
        return []
        
    print(f"Found {len(df)} entries for today")
    
    # --- Feature Engineering (Must match V28/V30 training) ---
    # Need history for rolling features
    
    # Get history for these dogs
    dog_ids = df['FastTrack_DogId'].unique()
    placeholders = ','.join(['?']*len(dog_ids))
    
    hist_query = f"""
    SELECT
        ge.GreyhoundID as FastTrack_DogId, rm.MeetingDate as date_dt,
        ge.FinishTime as RunTime, ge.Split as SplitMargin, ge.Position as Place,
        ge.BSP as StartPrice, ge.PrizeMoney as Prizemoney,
        r.Distance, t.TrackName as Track, ge.Box
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.GreyhoundID IN ({placeholders})
      AND rm.MeetingDate < ?
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
    ORDER BY ge.GreyhoundID, rm.MeetingDate
    """
    params = list(dog_ids) + [today_str]
    # Explicitly casting numpy ints to python ints for sqlite adapter safety
    params = [int(x) if hasattr(x, 'item') else x for x in list(dog_ids)] + [today_str]
    
    hist_df = pd.read_sql_query(hist_query, conn, params=params)
    print(f"[DEBUG] Loaded {len(hist_df)} historical rows for {len(dog_ids)} dogs.")
    
    conn.close()
    
    # Combine History + Today for feature calc
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    hist_df['date_dt'] = pd.to_datetime(hist_df['date_dt'])
    
    # Quick fix for numeric cols
    for col in ['Place', 'RunTime', 'SplitMargin', 'StartPrice', 'Prizemoney', 'Distance', 'Box']:
        hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Concat
    full_df = pd.concat([hist_df, df], ignore_index=True)
    full_df = full_df.sort_values(['FastTrack_DogId', 'date_dt'])
    
    # Basic Feature Prep
    full_df['StartPrice_probability'] = (1 / full_df['StartPrice']).fillna(0)
    full_df['Prizemoney_norm'] = np.log10(full_df['Prizemoney'].fillna(0) + 1) / 12
    full_df['Place_inv'] = (1 / full_df['Place']).fillna(0)
    full_df['Place_log'] = np.log10(full_df['Place'] + 1).fillna(0)
    full_df['BSP_log'] = np.log(full_df['StartPrice'].clip(lower=1.01)).fillna(0)
    
    # Track Stats
    win_df = full_df[full_df['Place'] == 1]
    median_win_time = win_df[win_df['RunTime'] > 0].groupby(['Track', 'Distance'])['RunTime'].median().reset_index()
    median_win_time.columns = ['Track', 'Distance', 'RunTime_median']
    median_win_split = win_df[win_df['SplitMargin'] > 0].groupby(['Track', 'Distance'])['SplitMargin'].median().reset_index()
    median_win_split.columns = ['Track', 'Distance', 'SplitMargin_median']
    
    median_win_time['speed_index'] = median_win_time['RunTime_median'] / median_win_time['Distance']
    scaler_speed = MinMaxScaler()
    # Fit only on available data, handle empty case
    if not median_win_time.empty:
        median_win_time['speed_index'] = scaler_speed.fit_transform(median_win_time[['speed_index']])
    else:
        median_win_time['speed_index'] = 0

    full_df = full_df.merge(median_win_time[['Track', 'Distance', 'RunTime_median', 'speed_index']], on=['Track', 'Distance'], how='left')
    full_df = full_df.merge(median_win_split, on=['Track', 'Distance'], how='left')
    
    full_df['RunTime_norm'] = (full_df['RunTime_median'] / full_df['RunTime']).clip(0.9, 1.1)
    full_df['RunTime_norm'] = full_df['RunTime_norm'].fillna(0)
    
    full_df['SplitMargin_norm'] = (full_df['SplitMargin_median'] / full_df['SplitMargin']).clip(0.9, 1.1)
    full_df['SplitMargin_norm'] = full_df['SplitMargin_norm'].fillna(0)
    
    # Speed feature (V28/V30 use this)
    full_df['Speed'] = full_df['Distance'] / full_df['RunTime']
    full_df['Speed'] = full_df['Speed'].fillna(0)

    box_win = full_df[full_df['Place']==1].groupby(['Track', 'Distance', 'Box'])['Place'].count().reset_index()
    box_total = full_df.groupby(['Track', 'Distance', 'Box'])['Place'].count().reset_index()
    
    # Safe Merge
    box_stats = pd.merge(box_total, box_win, on=['Track','Distance','Box'], how='left')
    box_stats['Box_Win_Pct'] = box_stats['Place_y'] / box_stats['Place_x']
    box_stats = box_stats[['Track','Distance','Box','Box_Win_Pct']].fillna(0)
    
    full_df = full_df.merge(box_stats, on=['Track', 'Distance', 'Box'], how='left')
    full_df['Box_Win_Pct'] = full_df['Box_Win_Pct'].fillna(0)
    
    # Rolling Features
    # V28 uses: RunTime_norm, SplitMargin_norm, Place_inv, Place_log, Prizemoney_norm, Speed
    # V30 uses: RunTime_norm, SplitMargin_norm, Place_inv, Place_log, Prizemoney_norm, Speed, BSP_log
    features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'Speed', 'BSP_log']
    aggregates = ['min', 'max', 'mean', 'median', 'std']
    rolling_windows = ['28D', '91D', '365D']
    
    dataset = full_df.set_index(['FastTrack_DogId', 'date_dt']).sort_index()
    
    for w in rolling_windows:
        rolling_res = (
            dataset.reset_index(level=0)
            .groupby('FastTrack_DogId')[features]
            .rolling(w)
            .agg(aggregates)
            .groupby(level=0)
            .shift(1)
        )
        agg_cols = [f"{f}_{a}_{w}" for f, a in itertools.product(features, aggregates)]
        dataset[agg_cols] = rolling_res
        
    dataset = dataset.fillna(0).reset_index()
    # pd.set_option('future.no_silent_downcasting', True)  # Removed: Causes error in recent pandas

    
    # Filter back to TODAY'S entries only
    # Match on EntryID to be safe, but we don't have EntryID in hist_df.
    # Match on date_dt == today_str and DogID
    today_dt = pd.to_datetime(today_str)
    model_df = dataset[dataset['date_dt'] == today_dt].copy()
    
    if len(model_df) == 0:
        print("Feature engineering resulted in empty set (no history match?).")
        return []

    print(f"Features generated for {len(model_df)} runners.")
    
    # Feature Debug
    has_history = False
    if len(model_df) > 0:
        sample = model_df.iloc[0]
        # Check if we have non-zero history features (e.g. rolling mean)
        # If 'RunTime_norm_mean_28D' is 0, likely no history.
        chk_col = 'RunTime_norm_mean_28D'
        if chk_col in sample and sample[chk_col] != 0:
             has_history = True
             
        if not has_history:
            print("\n[WARNING] ROLLING FEATURES ARE ZERO. DB likely lacks history for these dogs.")
            print("Predictions will be flat/inaccurate until race history is imported.")

    # Predict
    prob_v28 = predictor_v28.predict_proba(model_df)
    prob_v30 = predictor_v30.predict_proba(model_df)
    
    # Hybrid Prob
    p1_v28 = prob_v28[1] if 1 in prob_v28.columns else prob_v28.iloc[:, 1]
    p1_v30 = prob_v30[1] if 1 in prob_v30.columns else prob_v30.iloc[:, 1]
    
    model_df['prob_model'] = (p1_v28 + p1_v30) / 2
    
    # Normalize Prob per Race
    model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())
    
    # 4. Filter & Staking
    print("\n[4/4] Filtering and Staking (Hybrid + Eighth Kelly)...")
    
    # Strategy Params
    VALUE_THRESHOLD = 0.75
    PRICE_CAP = 6.0
    BANKROLL = 200.0
    MIN_BET = 1.0
    MAX_BET = 500.0
    
    def calculate_stake(row, bank):
        # Kelly = (bp - q) / b
        # price is the market price (StartingPrice from Betfair)
        market_price = float(row.get('StartingPrice', 0))
        if market_price <= 1.01: return 0
        
        b = market_price - 1
        p = row['prob_model']
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Eighth Kelly
        f = kelly_fraction * 0.125
        
        if f <= 0: return 0
        
        stake = bank * f
        return stake

    # Calculate Rated Price
    model_df['RatedPrice'] = 1 / model_df['prob_model']
    
    # Prepare Candidates List
    candidates = []
    
    for _, row in model_df.iterrows():
        # Get Live Price (StartingPrice)
        raw_price = row.get('StartingPrice')
        
        try:
            market_price = float(raw_price) if raw_price else 0.0
        except:
            market_price = 0.0
            
        rate_price = row['RatedPrice']
        prob = row['prob_model']
        
        # Debug top probabilities
        if prob > 0.15: # Only show decent chances
             print(f"[DEBUG-ROW] {row['GreyhoundName']} ({row['Track']}) Prob: {prob:.2f} Rated: ${rate_price:.2f} Market: ${market_price:.2f}")

        if market_price <= 1.01 or market_price > PRICE_CAP:
            continue
            
        # Value Filter
        if market_price > rate_price * (1 + VALUE_THRESHOLD):
            
            stake = calculate_stake(row, BANKROLL)
            stake = min(stake, MAX_BET) 
            
            if stake < MIN_BET:
                print(f"[DEBUG-SKIP] {row['GreyhoundName']} Stake too low: ${stake:.2f}")
                continue
                
            # Check Time
            try:
                rtime = str(row['RaceTime'])[:5]
            except:
                rtime = "00:00"
                
            candidates.append({
                'Date': today_str,
                'Track': row['Track'],
                'Race': row['RaceNumber'],
                'RaceTime': rtime,
                'Box': row['Box'],
                'Dog': row['GreyhoundName'],
                'Strategy': 'Hybrid V28/V30',
                'ModelProb': prob,
                'RatedPrice': rate_price,
                'MarketPrice': market_price,
                'Value': market_price / rate_price,
                'Stake': round(stake, 2)
            })
            
            print(f"[TIP] {row['GreyhoundName']} ({row['Track']} R{row['RaceNumber']}) "
                  f"Rated ${rate_price:.2f} Market ${market_price:.2f} "
                  f"Stake ${stake:.2f}")

    print(f"Generated {len(candidates)} tips.")
    return candidates

def sqlite3_connect(db_path):
    import sqlite3
    return sqlite3.connect(db_path)

if __name__ == "__main__":
    results = run_daily_predictions()
    if results:
        df = pd.DataFrame(results)
        print(df[['RaceTime', 'Track', 'Race', 'Dog', 'MarketPrice', 'Stake']])
