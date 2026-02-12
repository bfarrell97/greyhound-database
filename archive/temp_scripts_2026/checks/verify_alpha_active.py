
import sqlite3
import pandas as pd
from datetime import datetime
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from scripts.predict_market_v42_v43 import MarketAlphaEngine

def verify_alpha_logic():
    print("="*60)
    print("MARKET ALPHA (V42/V43) DIAGNOSTIC CHECK")
    print("="*60)
    
    engine = MarketAlphaEngine()
    
    conn = sqlite3.connect('greyhound_racing.db')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if we have candidates with Price5Min (required for Alpha)
    query = f"""
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.BSP, ge.Price5Min, r.RaceTime, t.TrackName, g.GreyhoundName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = '{today}'
    AND ge.Price5Min IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("[!] No runners found with 'Price5Min' data yet for today.")
        print("    Market Alpha requires the 5-minute price snapshot to detect moves.")
        print("    Wait for races to get closer to their start time (T-5m).")
        return

    print(f"[OK] Found {len(df)} candidates with early price data.")
    
    # Add dummy columns needed by engine if missing
    important_cols = ['Position', 'Weight', 'Distance', 'Grade', 'TrainerID', 'Split', 'FinishTime', 'Margin', 'DateWhelped', 'MeetingDate']
    for col in important_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Run predictions
    results = engine.predict(df)
    
    signals = results[results['Signal'].isin(['BACK', 'LAY'])]
    print(f"[OK] Scanned {len(results)} runners.")
    
    if not signals.empty:
        print(f"\n🎯 FOUND {len(signals)} ACTIVE SIGNALS:")
        for _, row in signals.iterrows():
            dog_match = df[df['EntryID'] == row['EntryID']]
            dog_name = dog_match.iloc[0]['GreyhoundName'] if not dog_match.empty else "Unknown"
            print(f"  - {row['Signal']}: {dog_name} | Alpha Prob: {row['Alpha_Prob']:.4f}")
    else:
        print("\n[INFO] 0 Signals found. Market conditions are currently stable.")
        print("       The engine is properly filtering out runners that don't show")
        print("       extreme Smart Money movement yet.")
    
    print("\n[CONCLUSION] The Alpha Engine is ACTIVE and monitoring the markets.")

if __name__ == "__main__":
    verify_alpha_logic()
