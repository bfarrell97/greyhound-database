import pandas as pd
import sqlite3
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    print("Could not import FeatureEngineerV41")
    sys.exit(1)

def debug_prediction():
    conn = sqlite3.connect('greyhound_racing.db')
    print("Loading data for 2025-12-26...")
    
    # SIMULATE APP FIX: 90 Day Window
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    today_str = '2025-12-26'
    
    print(f"Fetching data from {start_date} to today (90 Day Window)...")
    
    query = f"""
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
        ge.Split, ge.FinishTime, ge.Margin,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.GreyhoundName as Dog, g.DateWhelped, r.RaceTime, r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '{start_date}'
    """
    df_all = pd.read_sql_query(query, conn)
    
    if df_all.empty:
        print("No runners found!")
        conn.close()
        return

    print(f"Loaded {len(df_all)} runners (History + Today).")
    
    # Run Feature Engineering on FULL set
    fe = FeatureEngineerV41()
    print("\nRunning Feature Engineering on 90-DAY WINDOW...")
    df_feat = fe.calculate_features(df_all)
    
    # Filter for TODAY and HOLLY ROSE
    df_feat['MeetingDate_Str'] = pd.to_datetime(df_feat['MeetingDate']).dt.strftime('%Y-%m-%d')
    df_today = df_feat[df_feat['MeetingDate_Str'] == today_str]
    
    df_holly_fe = df_today[df_today['Dog'].str.contains('HOLLY ROSE', na=False)]
    
    if df_holly_fe.empty:
        print("HOLLY ROSE not found in today's results!")
    else:
        print("\nAfter FE Features (HOLLY ROSE):")
        cols = ['RunTimeNorm_Lag1', 'Box', 'Distance', 'Trainer_Track_Rate', 'Dog_Win_Rate', 'Split_Lag1'] 
        present_cols = [c for c in cols if c in df_holly_fe.columns]
        print(df_holly_fe[present_cols].to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    debug_prediction()
