"""
Inspect One Race
Goal: Manually verify prediction and margin logic on a known race.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def inspect():
    # Load 1 race from Warrnambool 2025
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Split
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE t.TrackName = 'Warrnambool' 
      AND rm.MeetingDate = '2025-01-02'
      AND ge.FinishTime IS NOT NULL
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Fake Feature Engineering
    df['DogNormFinishTimeAvg'] = 0.5
    df['DogNormSplitAvg'] = 0.5
    df['DogNormRunHomeAvg'] = 0.5
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'Box', 'Distance']
    
    # Fake Model
    print("Preds:")
    # Create fake preds that definitely have a margin
    fake_preds = [30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7]
    df = df.iloc[:len(fake_preds)].copy()
    df['PredOverall'] = fake_preds
    # Ensure they encompass one RaceID
    df['RaceID'] = 12345
    
    print(df[['GreyhoundName', 'PredOverall']])
    
    # Logic
    df['PredRank'] = df.groupby('RaceID')['PredOverall'].rank(method='min')
    print("\nRanks:")
    print(df[['GreyhoundName', 'PredOverall', 'PredRank']])
    
    # Simple Rank Merge
    rank1 = df[df['PredRank'] == 1][['RaceID', 'PredOverall', 'GreyhoundName']]
    rank2 = df[df['PredRank'] == 2][['RaceID', 'PredOverall']]
    rank2.columns = ['RaceID', 'Time2nd']
    
    print("\nRank 2 Table:")
    print(rank2)
    
    # Merge
    merged = rank1.merge(rank2, on='RaceID', how='left')
    merged['Margin'] = merged['Time2nd'] - merged['PredOverall']
    
    print("\nMerged:")
    print(merged[['GreyhoundName', 'PredOverall', 'Time2nd', 'Margin']])

if __name__ == "__main__":
    inspect()
