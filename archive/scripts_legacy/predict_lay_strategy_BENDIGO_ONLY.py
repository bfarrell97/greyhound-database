
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
print(f"DEBUG: GreyhoundMLModel loaded from: {src.models.ml_model.__file__}")

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
    now = datetime.now()
    # Broad scan: -2h to +36h to capture "Today's" full card even if some races passed, 
    # and tomorrow's early ones if needed. User wants "Today".
    # Let's focus on Today + Tomorrow morning.
    markets = fetcher.get_greyhound_markets(
        from_time=now - timedelta(hours=2),
        to_time=now + timedelta(hours=36)
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
            if 'bendigo' not in market.event.name.lower() and 'bendigo' not in market.market_name.lower():
                continue
                
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
                     # Try guess from name prefix
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
            
            # Construct Form Data Dict
            race_data = {
                'track_name': track_name,
                'date': race_date_str,
                'race_number': race_number,
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
    
    # 3. Predict
    print("\n[3/4] Running ML Model...")
    model = GreyhoundMLModel()
    try:
        # Load the Regression Model (trained on DogNormTimeAvg, Box, Distance)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'greyhound_model.pkl')
        if not os.path.exists(model_path):
             model_path = 'greyhound_model.pkl'
             
        print(f"Loading model from: {model_path}")
        model.load_model(model_path)
    except Exception as e:
        print(f"Error: Could not load model: {e}")
        return []

    # Predict for Today
    today_str = datetime.now().strftime('%Y-%m-%d')
    # Threshold 0.1 refers to Predicted Time Margin (Seconds) for Regression Model
    predictions = model.predict_race_winners_v2(today_str, confidence_threshold=0.1)
    
    # 4. Filter for Lay Candidates
    print("\n[4/4] Filtering candidates...")
    candidates = []
    
    if predictions.empty:
        print("No candidates found.")
        return []
        
    for _, row in predictions.iterrows():
        # Candidates are already filtered by ml_model.py (Margin > 0.1, Odds < 2.25)
        candidates.append({
             'Date': today_str,
             'Track': row['CurrentTrack'],
             'Race': row['RaceNumber'],
             'Dog': row['GreyhoundName'],
             'Margin': row['Margin'],
             'Odds': row['Odds'],
             'Strategy': 'Regression Lay (Mg > 0.1, Odds < 2.25)' 
        })
             
    print(f"Generated {len(candidates)} candidates.")
    return candidates

if __name__ == "__main__":
    results = run_daily_predictions()
    if results:
        df = pd.DataFrame(results)
        print(df)
