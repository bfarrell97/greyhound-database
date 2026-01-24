import pandas as pd
import numpy as np
import sqlite3
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from src.features.feature_engineering_v38 import FeatureEngineerV38

def check_bsp_features():
    print("Checking BSP Lag Feature Quality...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT ge.*, ge.Position as Place, rm.MeetingDate as date_dt, ge.Box as RawBox, r.RaceTime,
           t.TrackName as RawTrack
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP > 0 LIMIT 5000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV38()
    df, features = fe.engineer_features(df)
    
    print("\nFeature Summary:")
    for col in ['BSP_Lag1', 'BSP_Lag2', 'BSP_Avg_3']:
        if col in df.columns:
            missing = (df[col] == -1).sum()
            total = len(df)
            print(f"{col}: {missing} missing ({(missing/total)*100:.1f}%)")
            print(f"  Mean (excl missing): {df[df[col] != -1][col].mean():.2f}")
        else:
            print(f"{col}: NOT FOUND")

if __name__ == "__main__":
    check_bsp_features()
