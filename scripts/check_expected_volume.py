import pandas as pd
import joblib
import sqlite3
import sys
import os
from datetime import datetime, timedelta

# Add path to find 'src'
sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
    from scripts.predict_v44_prod import MarketAlphaEngine
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41
    from scripts.predict_v44_prod import MarketAlphaEngine

def check_volume():
    print("--- 📊 ESTIMATING EXPECTED DAILY VOLUME (Last 14 Days) ---")
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Get 6 Months Data (Jun - Nov 2025)
    start_str = '2025-06-01'
    end_str = '2025-11-30'
    
    print(f"Loading data from {start_str} to {end_str}...")
    
    query = f"""
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
        ge.Split, ge.FinishTime, ge.Margin,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped, g.GreyhoundName as Dog
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate BETWEEN '{start_str}' AND '{end_str}'
    AND ge.Price5Min > 0
    AND ge.BSP > 0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No recent data found.")
        return

    print(f"Loaded {len(df)} runners.")
    
    # Initialize Engine
    engine = MarketAlphaEngine()
    
    # Run Prediction
    print("Running Engine...")
    df_pred = engine.predict(df)
    
    # Count Signals
    total_back = len(df_pred[df_pred['Signal'] == 'BACK'])
    total_lay = len(df_pred[df_pred['Signal'] == 'LAY'])
    
    days = df_pred['MeetingDate'].nunique()
    
    print(f"\n--- RESULTS (Over {days} Days) ---")
    print(f"Total BACK Signals: {total_back}")
    print(f"Total LAY Signals:  {total_lay}")
    print(f"Total Bets:         {total_back + total_lay}")
    print("-" * 30)
    print(f"Daily AVG BACK:     {total_back / days:.1f}")
    print(f"Daily AVG LAY:      {total_lay / days:.1f}")
    print(f"Daily AVG TOTAL:    {(total_back + total_lay) / days:.1f}")
    
    # Check TAS Exclusion Effect
    tas_active = df_pred[df_pred['TrackName'].isin(['Launceston', 'Hobart', 'Devonport'])]
    tas_sigs = len(tas_active[tas_active['Signal'] != 'PASS'])
    if tas_sigs > 0:
        print(f"\n[WARN] Found {tas_sigs} signals on TAS tracks! Exclusion might be failing?")
    else:
        print("\n[OK] TAS Exclusion verified (0 signals).")

if __name__ == "__main__":
    check_volume()
