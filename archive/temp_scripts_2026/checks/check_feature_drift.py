import pandas as pd
import sqlite3
import joblib
import sys
import os
import xgboost as xgb

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    pass

def check_feature_drift():
    conn = sqlite3.connect('greyhound_racing.db')
    print("Loading data Oct vs Nov...")
    query = """
    SELECT * FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE rm.MeetingDate >= '2025-10-01' AND rm.MeetingDate <= '2025-11-30'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['Month'] = pd.to_datetime(df['MeetingDate']).dt.to_period('M')
    
    # Check key column nulls
    print("\n--- NULL COUNTS ---")
    cols_to_check = ['Price5Min', 'BSP', 'Split']
    # Filter only existing columns
    cols_to_check = [c for c in cols_to_check if c in df.columns]
    print(df.groupby('Month')[cols_to_check].apply(lambda x: x.isnull().mean()))
    
    # Check Rolling Calc integrity (Simulated)
    
    print("\n--- TRAINER ID STATS ---")
    if 'TrainerID' in df.columns:
        print(df.groupby('Month')['TrainerID'].nunique())
    
    print("\n--- TRACK NAME STATS ---")
    # Finding track name col
    track_col = [c for c in df.columns if 'TrackName' in c]
    if track_col:
        print(df.groupby('Month')[track_col[0]].nunique())
    
if __name__ == "__main__":
    check_feature_drift()
