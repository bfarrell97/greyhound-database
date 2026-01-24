
import sys
import os
import sqlite3
import pandas as pd
import joblib
import xgboost as xgb
import warnings
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering_v41 import FeatureEngineerV41
from src.integration.betfair_fetcher import BetfairOddsFetcher

warnings.filterwarnings('ignore')

DB_PATH = "greyhound_racing.db"
MODEL_V41 = "models/xgb_v41_final.pkl"
MODEL_V42 = "models/xgb_v42_steamer.pkl"
MODEL_V43 = "models/xgb_v43_drifter.pkl"

def check_race(track_filter, race_num):
    print(f"Checking signals for {track_filter} Race {race_num}...")
    
    # 1. FETCH LIVE PRICES & RUNNERS
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Failed to login to Betfair.")
        return

    markets = fetcher.get_greyhound_markets()
    selected_market = None
    
    # Fuzzy Match Track
    for m in markets:
        m_name = m.event.name.upper()
        if track_filter.upper() in m_name:
            # Check Race Number (e.g., "R1" or "Race 1")
            # Betfair market name usually "R1 300m Gr5" or similar, or event name has race?
            # Actually market name is usually the race name/grade. The 'market_name' attribute might be "R1 401m"
            if f"R{race_num}" in m.market_name or f"RACE {race_num}" in m.market_name.upper():
                selected_market = m
                break
    
    if not selected_market:
        print(f"Could not find market for {track_filter} Race {race_num}")
        fetcher.logout()
        return

    print(f"Found Market: {selected_market.event.name} - {selected_market.market_name} ({selected_market.market_start_time})")
    
    # Get Prices
    prices = fetcher.get_market_prices(selected_market.market_id)
    fetcher.logout()
    
    if not prices:
        print("No prices available.")
        return
        
    # Map SelectionID -> Price
    price_map = {}
    for sid, p_data in prices.items():
        price_map[sid] = p_data.get('back', 0)

    # 2. LOAD DB DATA FOR DOGS
    # We need to find the dogs in the DB to get their history features.
    # We can match by Name.
    conn = sqlite3.connect(DB_PATH)
    
    # Get runner names from market object
    runners = {r.selection_id: r.runner_name for r in selected_market.runners}
    
    live_df_rows = []
    
    for sid, name in runners.items():
        # Clean name (remove trap number)
        import re
        clean_name = re.sub(r'^\d+\.\s*', '', name).strip().upper()
        current_price = price_map.get(sid, 0)
        
        # Pull latest Entry for this dog to build context?
        # Actually, we need the FULL feature set. 
        # FeatureEngineer requires a dataframe with columns like 'Split', 'FinishTime', etc for HISTORY.
        # It calculates features based on PAST races. 
        # The 'Current' row needs to exist for the prediction.
        # If the race is in the future, it might be in the DB if imported?
        # Let's check if the user imported today's cards.
        
        query = f"""
        SELECT 
            ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
            ge.Weight, ge.TrainerID, r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
            g.GreyhoundName as Dog, g.DateWhelped
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE UPPER(g.GreyhoundName) = '{clean_name.replace("'", "''")}'
        AND rm.MeetingDate = DATE('now')
        """
        # Note: 'now' in sqlite might be UTC. Better to pass python date.
        today_str = datetime.now().strftime('%Y-%m-%d')
        query = query.replace("DATE('now')", f"'{today_str}'")
        
        df_dog = pd.read_sql_query(query, conn)
        
        if df_dog.empty:
            # Maybe local time issue? Try just matching Dog Name and taking the LAST (future) entry?
            # Or just warn.
            print(f"Warning: No DB entry for {clean_name} today. Import might be missing.")
            continue
            
        row = df_dog.iloc[0].to_dict()
        row['Price5Min'] = current_price # Inject LIVE PRICE
        row['BSP'] = 0
        row['Position'] = 0
        row['Split'] = 0
        row['FinishTime'] = 0
        row['Margin'] = 0
        live_df_rows.append(row)

    conn.close()
    
    if not live_df_rows:
        print("No runner data found in DB. Make sure you have imported today's races.")
        return

    df = pd.DataFrame(live_df_rows)
    
    # 3. FEATURES
    fe = FeatureEngineerV41()
    # calculates_features usually expects full history. 
    # V41 Feature Engineer needs access to DB to pull history for the dogs in the DF.
    # Does `calculate_features` do that? 
    # Inspecting FeatureEngineerV41... it takes `df` and assumes it's the full dataset or it queries inside?
    # Usually it iterates the dataframe rows.
    # To get history features (Rolling Avg etc), the passed DF needs to contain the history OR the class needs to fetch it.
    # Let's assume the standard `calculate_features` works if we initialize it correctly or if it executes queries.
    # WAIT: The snippet I saw earlier for `calculate_features` (in older logs) seemed to calculate on the passed DF. 
    # If I only pass today's rows, Lag features will be NaN!
    # I need to fetch HISTORY for these dogs.
    
    # Re-strategy: Get history for these dogs from DB.
    # We can use `fe.load_data` style or just manually pull last 10 races for each dog + today's race.
    
    print("Calculating features (might take a moment)...")
    # Quick hack: We use the existing logic in FE if possible, but simpler:
    # Just run the V41 calculation. If it returns NaNs for Lags, the prediction will be poor.
    # To do it properly "on the fly" is hard without pre-computed store.
    
    # Lets try to rely on the fact that maybe the user has history?
    # Actually, for a single race, we can pull all history for the 8 dogs.
    dog_ids = df['GreyhoundID'].unique()
    conn = sqlite3.connect(DB_PATH)
    ids_str = ",".join([str(x) for x in dog_ids])
    hist_query = f"""
        SELECT 
            ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
            ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.Margin, ge.TrainerID,
            ge.Split, ge.FinishTime,
            r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
            g.GreyhoundName as Dog, g.DateWhelped
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        WHERE ge.GreyhoundID IN ({ids_str})
        ORDER BY rm.MeetingDate ASC
    """
    df_hist = pd.read_sql_query(hist_query, conn)
    conn.close()
    
    # Combine: Update the 'Today' rows in df_hist with the Live Prices from `df`
    # The `df` rows (from Step 2) are essentially duplicates of the last rows in `df_hist` (if imported), 
    # but `df_hist` might not have the live price.
    # Let's merge or update.
    
    for idx, row in df.iterrows():
        # Find this entry in hist
        mask = (df_hist['EntryID'] == row['EntryID'])
        if mask.any():
            df_hist.loc[mask, 'Price5Min'] = row['Price5Min']
    
    # Now run FE on full history
    df_features = fe.calculate_features(df_hist)
    
    # Filter to TODAY's race only
    target_race_id = df['RaceID'].iloc[0] # Assume duplicates share RaceID
    df_final = df_features[df_features['RaceID'] == target_race_id].copy()
    
    # 4. PREDICT
    model_v41 = joblib.load(MODEL_V41)
    model_v42 = joblib.load(MODEL_V42)
    model_v43 = joblib.load(MODEL_V43)
    
    # V41 Prob
    cols = fe.get_feature_list()
    for c in cols:
        if c not in df_final.columns: df_final[c] = 0
    
    dmatrix = xgb.DMatrix(df_final[cols])
    df_final['V41_Prob'] = model_v41.predict(dmatrix)
    df_final['ModelPrice'] = 1.0 / df_final['V41_Prob']
    
    # Discrepancy
    df_final['Discrepancy'] = df_final['Price5Min'] / df_final['ModelPrice']
    df_final['Price_Diff'] = df_final['Price5Min'] - df_final['ModelPrice']
    
    # V42/43
    # Use standard feature mapping (ensure columns match training)
    alpha_cols = ['Price5Min', 'V41_Prob', 'Discrepancy', 'Price_Diff', 'Box', 'Distance', 'RunTimeNorm_Lag1', 'Trainer_Track_Rate']
    # Ensure all present
    for c in alpha_cols:
        if c not in df_final.columns: df_final[c] = 0
        
    df_final['Steam_Prob'] = model_v42.predict_proba(df_final[alpha_cols])[:, 1]
    df_final['Drift_Prob'] = model_v43.predict_proba(df_final[alpha_cols])[:, 1]
    
    # 5. OUTPUT
    print("\n" + "="*80)
    print(f"SIGNALS: {track_filter.upper()} RACE {race_num}")
    print("="*80)
    print(f"{'DOG':<20} | {'PRICE':<6} | {'MODEL':<6} | {'STEAM%':<7} | {'DRIFT%':<7} | {'SIGNAL':<6}")
    print("-" * 80)
    
    for _, row in df_final.iterrows():
        p = row['Price5Min']
        mp = row['ModelPrice']
        s_prob = row['Steam_Prob']
        d_prob = row['Drift_Prob']
        
        sig = "-"
        # Back Logic
        thresh_back = 0.99
        if p < 2.0: thresh_back = 0.60
        elif p < 6.0: thresh_back = 0.55
        elif p < 10.0: thresh_back = 0.60
        elif p <= 40.0: thresh_back = 0.70
        
        if s_prob >= thresh_back: sig = "BACK"
        
        # Lay Logic
        thresh_lay = 0.65
        if p < 4.0: thresh_lay = 0.55
        elif p < 8.0: thresh_lay = 0.60
        
        if d_prob >= thresh_lay and p < 30.0:
            if sig == "BACK": sig = "CONFLICT"
            else: sig = "LAY"
            
        print(f"{row['Dog']:<20} | ${p:<5.2f} | ${mp:<5.2f} | {s_prob:.1%}   | {d_prob:.1%}   | {sig}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--race", type=int, required=True)
    args = parser.parse_args()
    
    check_race(args.track, args.race)
