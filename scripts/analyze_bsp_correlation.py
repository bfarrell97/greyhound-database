import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v40 import FeatureEngineerV40
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v40 import FeatureEngineerV40

def analyze_bsp():
    print("Loading Jan 2025 Data...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin,
        r.Distance, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2025-01-01'
    AND ge.BSP > 1.0
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Feature Engineering
    fe = FeatureEngineerV40()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    # Predict
    model = joblib.load('models/xgb_v40_hybrid.pkl')
    dtest = xgb.DMatrix(df[features])
    df['ModelProb'] = model.predict(dtest)
    df['ModelPrice'] = 1.0 / df['ModelProb']
    df['ImpliedProb'] = 1.0 / df['BSP']
    
    # metrics
    correlation = df[['ModelProb', 'ImpliedProb']].corr().iloc[0, 1]
    
    # Error metrics
    df['Diff'] = abs(df['ModelProb'] - df['ImpliedProb'])
    mae = df['Diff'].mean()
    
    # Favorites Alignment Logic
    market_faves = df.loc[df.groupby('RaceID')['ImpliedProb'].idxmax()]
    model_faves = df.loc[df.groupby('RaceID')['ModelProb'].idxmax()]
    
    comparison = pd.merge(
        market_faves[['RaceID', 'GreyhoundID', 'BSP', 'win']],
        model_faves[['RaceID', 'GreyhoundID', 'ModelPrice', 'win']],
        on='RaceID', suffixes=('_Mkt', '_Mod')
    )
    
    total_races = len(comparison)
    match_count = (comparison['GreyhoundID_Mkt'] == comparison['GreyhoundID_Mod']).sum()
    match_pct = (match_count / total_races) * 100 if total_races > 0 else 0
    
    mkt_wins = comparison[comparison['win_Mkt'] == 1].shape[0]
    mod_wins = comparison[comparison['win_Mod'] == 1].shape[0]

    # Write to file
    with open('bsp_analysis.txt', 'w') as f:
        f.write("\n---------------------------------------------------\n")
        f.write("MARKET PREDICTION ANALYSIS (Jan 2025)\n")
        f.write("---------------------------------------------------\n")
        f.write(f"Correlation (Model vs BSP): {correlation:.4f}\n")
        f.write(f"Mean Absolute Error (Prob): {mae:.4f}\n")
        f.write(f"Interpretation: {correlation > 0.6 and 'Strong Match' or 'Weak Match'}\n\n")
        
        f.write("Favorites Alignment:\n")
        f.write(f"Total Races: {total_races}\n")
        f.write(f"Model picked same Favorite as Market: {match_count} ({match_pct:.1f}%)\n")
        f.write(f"Market Favorite Win Rate: {mkt_wins}/{total_races} ({mkt_wins/total_races*100:.1f}%)\n")
        f.write(f"Model Favorite Win Rate:  {mod_wins}/{total_races} ({mod_wins/total_races*100:.1f}%)\n")
        
        if mod_wins > mkt_wins:
            f.write(">> MODEL IS SUPERIOR TO MARKET FAVORITES <<\n")
        else:
            f.write(">> Market Favorites are slightly better (Expected) <<\n")
            
    print("Analysis written to bsp_analysis.txt")

if __name__ == "__main__":
    analyze_bsp()
