
"""
Predict Lay Strategy Candidates (Odds-On Layer)
Uses direct Betfair Market Scanning to discover races and runners.
Populates DB with 'Race Day' data, then runs ML Model.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.betfair_fetcher import BetfairOddsFetcher
from src.core.database import GreyhoundDatabase
from src.models.ml_model import GreyhoundMLModel
import src.models.ml_model
# print(f"DEBUG: GreyhoundMLModel loaded from: {src.models.ml_model.__file__}")

def run_daily_predictions():
    """
    Run daily predictions pipeline:
    1. Scan Betfair for upcoming races (Next 36h)
    2. Import runners/races to DB
    3. Run ML Model
    4. Filter for Lay candidates
    """
    print("="*80)
    print(f"RUNNING DAILY PREDICTIONS (BETFAIR API): {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    # Initialize
    db = GreyhoundDatabase()
    fetcher = BetfairOddsFetcher()
    
    if not fetcher.login():
        print("Fatal: Could not login to Betfair.")
        return []

    # 1. Get Markets
    print("\n[1/4] Scanning Betfair Markets...")
    # Use UTC via betfair_fetcher default or explicit UTC
    # Passing None lets fetcher use its internal UTC logic which is safer
    markets = fetcher.get_greyhound_markets(
        from_time=None, # Use fetcher default (-2h to +24h UTC)
        to_time=None
    )
    
    if not markets:
        print("No markets found.")
        fetcher.logout()
        return []
        
    print(f"Found {len(markets)} markets.")
    
    # 2. Process & Import to DB
    print("\n[2/4] Importing Race Data to Database...")
    
    # Group by Meeting to be efficient/clean (though import handles individual races)
    imported_count = 0
    
    for market in markets:
        try:
            # Parse Event Name "Track (State) Date" -> e.g. "Taree (NSW) 13th Dec"
            event_name = market.event.name
            market_name = market.market_name # "R1 300m Mdn"
            market_start_time = market.market_start_time # datetime object
            
            # Extract Track
            # Remove date part if possible, or just take the first part
            # Regex to grab "Taree" from "Taree (NSW) 13th Dec"
            # Usually: "Name (State) DDth Mon"
            track_match = re.search(r'^([^(]+)', event_name)
            track_name = track_match.group(1).strip() if track_match else event_name
            
            # Format Date
            race_date_str = market_start_time.strftime('%Y-%m-%d')
            
            # Extract Race Number
            # Search "R1", "Race 1" in market name
            race_num_match = re.search(r'R(\d+)', market_name)
            if not race_num_match:
                race_num_match = re.search(r'Race\s+(\d+)', market_name)
            
            race_number = int(race_num_match.group(1)) if race_num_match else 0
            
            if race_number == 0:
                continue # Skip non-race markets
                
            # Extract Distance
            dist_match = re.search(r'(\d+)m', market_name)
            distance = int(dist_match.group(1)) if dist_match else 0
            
            # Extract Grade
            grade = market_name.replace(f"R{race_number}", "").replace(f"{distance}m", "").strip()
            
            # Fetch Live Odds for this Market
            market_odds = fetcher.get_market_odds(market.market_id)
            
            # Construct Form Data Dict
            entries = []
            for runner in market.runners:
                # if runner.status != 'ACTIVE':
                #    continue
                    
                runner_name = runner.runner_name
                # Clean name "1. Dog Name" -> "Dog Name"
                clean_name = re.sub(r'^\d+\.\s*', '', runner_name)
                
                # Get Box/Trap
                box = None
                if runner.metadata and 'TRAP' in runner.metadata:
                    box = runner.metadata['TRAP']
                
                if not box:
                     # Try guess from name prefix (e.g. "1. Dog Name")
                     prefix_match = re.match(r'^(\d+)\.', runner_name)
                     if prefix_match:
                         box = prefix_match.group(1)

                # Get Trainer (if available in metadata)
                trainer = ""
                if runner.metadata and 'TRAINER_NAME' in runner.metadata:
                    trainer = runner.metadata['TRAINER_NAME']

                # Get Live Odds
                price = market_odds.get(runner.selection_id)
                    
                entries.append({
                    'greyhound_name': clean_name,
                    'box': int(box) if box else None,
                    'trainer': trainer,
                    'owner': '', # Not in Betfair API
                    'sire': '',
                    'dam': '',
                    'starts': 0, 
                    'wins': 0,
                    'prizemoney': 0,
                    'starting_price': price # Inject Live Price
                })
            
            # Extract Time
            market_time = market.market_start_time
            race_time_str = ""
            if market_time:
                # Format to HH:MM (e.g. 2025-12-13 10:30:00 -> 10:30)
                try:
                    if isinstance(market_time, str):
                        dt = datetime.fromisoformat(market_time.replace('Z', '+00:00'))
                    else:
                        # Ensure UTC if naive (betfair usually returns naive UTC)
                        dt = market_time
                        if dt.tzinfo is None:
                            from datetime import timezone
                            dt = dt.replace(tzinfo=timezone.utc)
                    
                    # Convert to Local Time
                    local_dt = dt.astimezone()
                    race_time_str = local_dt.strftime('%H:%M')
                    
                except:
                    race_time_str = str(market_time)

            # Construct Form Data Dict
            race_data = {
                'track_name': track_name,
                'date': race_date_str,
                'race_number': race_number,
                'race_time': race_time_str,
                'race_name': market_name,
                'grade': grade,
                'distance': distance,
                'prize_money': '',
                'entries': entries
            }
            
            # Import
            print(f"[DEBUG] Importing {track_name} R{race_number} with {len(entries)} entries...")
            if db.import_form_guide_data(race_data, race_date_str, track_name):
                print(f"[DEBUG] SUCCESS Importing {track_name} R{race_number}")
                imported_count += 1
            else:
                print(f"[ERROR] FAILED Importing {track_name} R{race_number}")
                
        except Exception as e:
            print(f"Error processing market {market.market_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Imported {imported_count} races into DB.")
    fetcher.logout()
    
    # 3. Predict using PACE MODEL (V1)
    print("\n[3/4] Running Pace Model...")
    import pickle
    
    pace_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pace_xgb_model.pkl')
    if not os.path.exists(pace_model_path):
        print(f"Error: Pace model not found at {pace_model_path}")
        return []
    
    print(f"Loading pace model from: {pace_model_path}")
    with open(pace_model_path, 'rb') as f:
        pace_model = pickle.load(f)
    
    # Get today's data from DB
    today_str = datetime.now().strftime('%Y-%m-%d')
    db = GreyhoundDatabase()
    conn = db.get_connection()
    
    # Load benchmarks
    bench_df = pd.read_sql_query("SELECT TrackName, Distance, MedianTime AS TrackDistMedian FROM Benchmarks", conn)
    
    # Get today's entries with history
    query = """
    SELECT ge.EntryID, ge.RaceID, g.GreyhoundID, g.GreyhoundName, 
           r.RaceNumber, r.RaceTime, r.Distance, t.TrackName,
           ge.Box, ge.StartingPrice, rm.MeetingDate
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate = ?
    """
    today_df = pd.read_sql_query(query, conn, params=(today_str,))
    
    if len(today_df) == 0:
        print(f"No entries found for {today_str}")
        conn.close()
        return []
    
    print(f"Found {len(today_df)} entries for today")
    
    # Get historical NormTime for each dog
    history_query = """
    SELECT ge.GreyhoundID, rm.MeetingDate, t.TrackName, r.Distance, ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID  
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.GreyhoundID IN ({}) 
      AND rm.MeetingDate < ?
      AND ge.FinishTime IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate
    """.format(','.join(str(x) for x in today_df['GreyhoundID'].unique()))
    
    hist_df = pd.read_sql_query(history_query, conn, params=(today_str,))
    conn.close()
    
    # Calculate NormTime for history
    hist_df = hist_df.merge(bench_df, on=['TrackName', 'Distance'], how='left')
    hist_df['NormTime'] = hist_df['FinishTime'] - hist_df['TrackDistMedian']
    hist_df = hist_df.dropna(subset=['NormTime'])
    
    # Calculate rolling features per dog
    hist_df = hist_df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = hist_df.groupby('GreyhoundID')
    hist_df['Lag1'] = g['NormTime'].shift(0)  # Most recent
    hist_df['Lag2'] = g['NormTime'].shift(1)
    hist_df['Lag3'] = g['NormTime'].shift(2)
    hist_df['Roll3'] = g['NormTime'].transform(lambda x: x.rolling(3, min_periods=3).mean())
    hist_df['Roll5'] = g['NormTime'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    hist_df['PrevDate'] = g['MeetingDate'].shift(0)
    
    # Get latest features per dog
    latest = hist_df.groupby('GreyhoundID').last().reset_index()
    latest = latest[['GreyhoundID', 'Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'PrevDate']]
    
    # Merge with today's entries
    today_df = today_df.merge(latest, on='GreyhoundID', how='left')
    today_df['MeetingDate'] = pd.to_datetime(today_df['MeetingDate'])
    today_df['PrevDate'] = pd.to_datetime(today_df['PrevDate'])
    today_df['DaysSince'] = (today_df['MeetingDate'] - today_df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    today_df['Box'] = pd.to_numeric(today_df['Box'], errors='coerce').fillna(0)
    today_df['Distance'] = pd.to_numeric(today_df['Distance'], errors='coerce').fillna(0)
    
    # Drop dogs without enough history (need Roll5)
    today_df = today_df.dropna(subset=['Roll5'])
    print(f"Dogs with 5+ prior races: {len(today_df)}")
    
    if len(today_df) == 0:
        return []
    
    # Predict pace
    X = today_df[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']]
    today_df['PredPace'] = pace_model.predict(X)
    
    # Rank within each race and calculate gap
    today_df = today_df.sort_values(['RaceID', 'PredPace'])
    today_df['Rank'] = today_df.groupby('RaceID').cumcount() + 1
    today_df['NextPace'] = today_df.groupby('RaceID')['PredPace'].shift(-1)
    today_df['Gap'] = today_df['NextPace'] - today_df['PredPace']
    
    # Filter: Rank 1 only (pace leaders)
    leaders = today_df[today_df['Rank'] == 1].copy()
    
    # 4. Apply BACK Strategy Filters
    print("\n[4/4] Filtering for Back Strategy...")
    
    # Convert odds
    def safe_convert_odds(x):
        if pd.notna(x) and x not in [0, '0', 'None', None]:
            try:
                return float(x)
            except:
                return 100.0
        return 100.0
    
    leaders['Odds'] = leaders['StartingPrice'].apply(safe_convert_odds)
    
    # Back Strategy Filters:
    # - Gap >= 0.15 (clear pace advantage)
    # - Middle Distance: 400-550m
    # - Odds: $3-$8
    candidates_df = leaders[
        (leaders['Gap'] >= 0.15) &
        (leaders['Distance'] >= 400) & (leaders['Distance'] < 550) &
        (leaders['Odds'] >= 3.0) & (leaders['Odds'] <= 8.0)
    ].copy()
    
    print(f"Found {len(candidates_df)} Back candidates (Gap>=0.15, Dist 400-550m, Odds $3-$8)")
    
    candidates = []
    for _, row in candidates_df.iterrows():
        candidates.append({
             'Date': today_str,
             'Track': row['TrackName'],
             'Race': row['RaceNumber'],
             'RaceTime': row['RaceTime'],
             'Box': row['Box'],
             'Dog': row['GreyhoundName'],
             'Margin': row['Gap'],
             'Odds': row['Odds'],
             'Strategy': 'Pace Back'
        })
        print(f"[DEBUG] Candidate: {row['GreyhoundName']} Gap: {row['Gap']:.2f} Odds: ${row['Odds']:.2f}")
             
    print(f"Generated {len(candidates)} candidates.")
    return candidates

if __name__ == "__main__":
    results = run_daily_predictions()
    if results:
        df = pd.DataFrame(results)
        print(df)
