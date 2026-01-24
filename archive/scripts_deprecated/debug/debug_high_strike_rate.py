"""
Debug High Strike Rate (Survivor Bias Check)
Goal: Check if Goulburn/Townsville are missing data for losers (e.g. Missing Splits), 
causing the model to only see/bet on winners.
"""
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'greyhound_racing.db'

def inspect_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        t.TrackName,
        r.RaceID,
        ge.Position,
        ge.Split,
        ge.FinishTime
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2025-01-01'
      AND t.TrackName IN ('Goulburn', 'Townsville', 'Sandown Park')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 1. Check Completeness per Race
    print("\n1. DATA COMPLETENESS (Rows per Race)")
    print("-" * 60)
    race_counts = df.groupby(['TrackName', 'RaceID']).size().reset_index(name='DogCount')
    print(race_counts.groupby('TrackName')['DogCount'].describe()[['count', 'mean', 'min', 'max']])
    
    # 2. Check Missing Splits for Losers vs Winners
    df['IsWin'] = (df['Position'] == '1')
    df['HasSplit'] = df['Split'].notna() & (df['Split'] != '')
    
    print("\n2. SPLIT AVAILABILITY (Winner vs Loser)")
    print("-" * 60)
    
    stats = df.groupby(['TrackName', 'IsWin'])['HasSplit'].agg(['count', 'sum', 'mean'])
    stats.columns = ['TotalDogs', 'DogsWithSplit', 'PctWithSplit']
    print(stats)
    
    # Check if 'DogsWithSplit' count roughly equals 'TotalDogs' for losers?
    # If PctWithSplit for Losers is low (< 0.5), we have a problem.

if __name__ == "__main__":
    inspect_data()
