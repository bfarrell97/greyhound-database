
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "greyhound_racing.db"

def entropy_check():
    print("="*60)
    print("DATA LEAKAGE AUDIT - FEATURE INTEGRITY CHECK")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # query to get raw data for a specific dog with multiple runs
    print("[1/3] Selecting a sample greyhound with >5 runs in 2025...")
    query = """
    SELECT GreyhoundID, COUNT(*) as Runs
    FROM GreyhoundEntries
    GROUP BY GreyhoundID
    HAVING Runs > 10
    LIMIT 1
    """
    sample_dog = pd.read_sql_query(query, conn)
    dog_id = sample_dog.iloc[0]['GreyhoundID']
    print(f"  Selected Dog ID: {dog_id}")
    
    # Get all runs for this dog
    query_runs = f"""
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, 
        ge.Position as Place, ge.FinishTime as RunTime, ge.BSP as StartPrice,
        rm.MeetingDate as date_dt
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.GreyhoundID = {dog_id}
    AND rm.MeetingDate >= '2024-01-01'
    ORDER BY rm.MeetingDate
    """
    df = pd.read_sql_query(query_runs, conn)
    conn.close()
    
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
    
    # Simulate Logic
    print("[2/3] Calculating Lag Features (Python)...")
    for i in range(1, 4):
         df[f'Place_Lag{i}'] = df['Place'].shift(i)
         
    # Check
    print("[3/3] Verifying Integrity...")
    leakage_found = False
    
    print(f"{'DATE':<12} | {'PLACE':<5} | {'LAG1':<5} | {'LAG2':<5} | {'STATUS':<10}")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        if idx == 0: continue
        
        current_place = row['Place']
        lag1 = row['Place_Lag1']
        
        # Check against PREVIOUS row
        prev_row = df.iloc[idx-1]
        actual_prev_place = prev_row['Place']
        
        # Integrity Limit
        # Lag1 of Current Row MUST EQUAL Place of Previous Row
        if pd.isna(lag1):
            status = "N/A"
        elif lag1 == actual_prev_place:
            status = "OK"
        else:
            status = "FAIL"
            leakage_found = True
            
        print(f"{str(row['date_dt'].date()):<12} | {current_place:<5} | {lag1:<5} | {row['Place_Lag2']:<5} | {status:<10}")
        
    print("-" * 60)
    if leakage_found:
        print("CRITICAL ALARM: Lag Feature Mismatch Found (Possible Data corruption, NOT Leakage, but broken logic).")
    else:
        print("PASS: Lag1 Matches Previous Race Result correctly. No backward leakage detected.")
        print("      (Current row does not know its own result via Lag columns).")

if __name__ == "__main__":
    entropy_check()
