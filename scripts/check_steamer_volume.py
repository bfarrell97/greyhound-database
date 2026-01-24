import pandas as pd
import sqlite3
import joblib
import xgboost as xgb
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def check_steamer_volume():
    print("Checking Steamer Heuristic Volume...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2025-01-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    
    # Predict
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df_clean[features])
    df_clean['Prob'] = model.predict(dtest)
    df_clean['ImpliedProb'] = 1.0 / df_clean['BSP']
    df_clean['Edge'] = df_clean['Prob'] - df_clean['ImpliedProb']
    
    # Heuristic
    # PredSteamer: Edge < -0.10 & Prob > 0.20
    steamers = df_clean[
        (df_clean['Edge'] < -0.10) &
        (df_clean['Prob'] > 0.20)
    ].copy()
    
    count = len(steamers)
    start_date = df_clean['MeetingDate'].min()
    end_date = df_clean['MeetingDate'].max()
    days = (end_date - pd.to_datetime(start_date)).days + 1
    
    print("\n" + "="*60)
    print("STEAMER STRATEGY VOLUME")
    print(f"Filter: Edge < -0.10 & Prob > 0.20")
    with open('volume.txt', 'w') as f:
        f.write(f"Total Bets (2025): {count}\n")
        f.write(f"Days: {days}\n")
        f.write(f"Bets Per Day: {count/days:.1f}\n")
        f.write(f"Bets Per Month: {count/days*30:.1f}\n")
    
    print("Volume written to volume.txt")

if __name__ == "__main__":
    check_steamer_volume()
