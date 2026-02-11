"""
Diagnostic script to check why signals aren't being generated.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')

# Load the prediction engine
from scripts.predict_v44_prod import MarketAlphaEngine

print("Initializing Engine...")
engine = MarketAlphaEngine(db_path='greyhound_racing.db')

# Get today's data
conn = sqlite3.connect('greyhound_racing.db')
today = datetime.now().strftime('%Y-%m-%d')

query = """
SELECT 
    ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
    ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
    ge.Split, ge.FinishTime, ge.Margin,
    r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
    g.DateWhelped, g.GreyhoundName as Dog, r.RaceTime
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate = ?
AND ge.Price5Min IS NOT NULL
AND ge.Price5Min > 0
AND ge.Price5Min < 15
"""

df = pd.read_sql_query(query, conn, params=[today])
conn.close()

print(f"\nLoaded {len(df)} runners with Price5Min < $15 for today")

if df.empty:
    print("ERROR: No data to predict on!")
else:
    # Run predictions
    print("\nRunning predictions...")
    results = engine.predict(df, use_cache=True)
    
    print(f"\nPrediction Results:")
    print(f"Total runners: {len(results)}")
    
    # Check Steam_Prob distribution
    print(f"\n--- Steam_Prob (BACK Signal) Distribution ---")
    print(f"Min: {results['Steam_Prob'].min():.3f}")
    print(f"Max: {results['Steam_Prob'].max():.3f}")
    print(f"Mean: {results['Steam_Prob'].mean():.3f}")
    print(f"Median: {results['Steam_Prob'].median():.3f}")
    
    # Count above threshold
    back_threshold = 0.35
    back_signals = results[results['Steam_Prob'] >= back_threshold]
    print(f"\n> {back_threshold} threshold: {len(back_signals)} signals")
    
    if not back_signals.empty:
        print("\nTop 5 BACK signals:")
        top_back = back_signals.nlargest(5, 'Steam_Prob')[['TrackName', 'Dog', 'RaceTime', 'Steam_Prob', 'Price5Min']]
        print(top_back.to_string())
    
    # Check Drift_Prob distribution
    if 'Drift_Prob' in results.columns:
        print(f"\n--- Drift_Prob (LAY Signal) Distribution ---")
        print(f"Min: {results['Drift_Prob'].min():.3f}")
        print(f"Max: {results['Drift_Prob'].max():.3f}")
        print(f"Mean: {results['Drift_Prob'].mean():.3f}")
        print(f"Median: {results['Drift_Prob'].median():.3f}")
        
        lay_threshold = 0.60
        lay_signals = results[results['Drift_Prob'] >= lay_threshold]
        print(f"\n> {lay_threshold} threshold: {len(lay_signals)} signals")
        
        if not lay_signals.empty:
            print("\nTop 5 LAY signals:")
            top_lay = lay_signals.nlargest(5, 'Drift_Prob')[['TrackName', 'Dog', 'RaceTime', 'Drift_Prob', 'Price5Min']]
            print(top_lay.to_string())
    
    # Show Signal column summary
    if 'Signal' in results.columns:
        print(f"\n--- Signal Summary ---")
        print(results['Signal'].value_counts())
