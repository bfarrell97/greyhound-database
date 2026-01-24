import pandas as pd
import sqlite3
import joblib
import sys
import os
from datetime import datetime

# Add path to find 'src'
sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
    # Mocking the engine class or importing it if separate? 
    # Let's import the script directly if possible, or copy logic.
    # Importing is cleaner.
    from scripts.predict_v44_prod import MarketAlphaEngine
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from scripts.predict_v44_prod import MarketAlphaEngine

def debug_today():
    print("Loading Today's Data for Debugging...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Get today strings
    today = datetime.now().strftime('%Y-%m-%d')
    # today = '2025-12-29' # Hardcoded for safety if needed
    
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Price5Min, 
        ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = ?
    AND ge.Price5Min > 0
    """
    df = pd.read_sql_query(query, conn, params=(today,))
    conn.close()
    
    if len(df) == 0:
        print("No data found for today.")
        return

    print(f"Loaded {len(df)} runners for {today}.")
    
    # Initialize Engine
    engine = MarketAlphaEngine()
    
    # Run Prediction
    print("Running Prediction Engine...")
    df_pred = engine.predict(df)
    
    # Analyze Signals
    back_signals = df_pred[df_pred['Signal'] == 'BACK'].copy()
    
    print(f"\n=== ALPHA ENGINE (BACK) SIGNALS FOR {today} ===")
    print(f"Total Found: {len(back_signals)}")
    
    if len(back_signals) > 0:
        # Sort by Time
        back_signals = back_signals.sort_values(['RaceTime', 'TrackName'])
        print(back_signals[['RaceTime', 'TrackName', 'Box', 'Dog', 'Price5Min', 'Steam_Prob', 'Signal']].to_string(index=False))
    else:
        print("No BACK signals found.")
        
    # Deep Dive: Near Misses
    print("\n--- NEAR MISSES (BACK) ---")
    # Show candidates with Prob > 0.30 (Threshold is 0.38)
    near_backs = df_pred[(df_pred['Steam_Prob'] > 0.30) & (df_pred['Signal'] != 'BACK')]
    near_backs = near_backs.sort_values('Steam_Prob', ascending=False).head(10)
    if len(near_backs) > 0:
        print(near_backs[['TrackName', 'Box', 'Price5Min', 'Steam_Prob', 'Signal']].to_string(index=False))
        
    print("\n--- NEAR MISSES (LAY) ---")
    # Show candidates with Prob > 0.50 (Threshold is 0.63)
    near_lays = df_pred[(df_pred['Drift_Prob'] > 0.50) & (df_pred['Signal'] != 'LAY')]
    # Filter out blocked TAS manually for clarity if needed
    # (Engine likely set them to PASS already)
    near_lays = near_lays.sort_values('Drift_Prob', ascending=False).head(10)
    if len(near_lays) > 0:
        print(near_lays[['TrackName', 'Box', 'Price5Min', 'Drift_Prob', 'Signal']].to_string(index=False))

if __name__ == "__main__":
    debug_today()
