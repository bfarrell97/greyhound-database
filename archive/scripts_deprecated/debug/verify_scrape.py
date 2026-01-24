
import sqlite3
import pandas as pd
import sys

# Removed sys.stdout.reconfigure

DB_PATH = 'greyhound_racing.db'
TARGET_DATE = '2025-12-10'

def get_fill_rates(conn, date):
    print(f"DEBUG: Checking {date}...")
    try:
        # Get meetings
        meetings = pd.read_sql_query(f"SELECT MeetingID FROM RaceMeetings WHERE MeetingDate = '{date}'", conn)
        if meetings.empty:
            print(f"DEBUG: No meetings for {date}")
            return None, 0
        
        meeting_ids = tuple(meetings['MeetingID'].tolist())
        if len(meeting_ids) == 1: meeting_ids = f"({meeting_ids[0]})"
        
        # Get races checks (PrizeMoney)
        races_df = pd.read_sql_query(f"SELECT RaceID, PrizeMoney FROM Races WHERE MeetingID IN {meeting_ids}", conn)
        if races_df.empty:
            print(f"DEBUG: No races for {date}")
            return None, 0
            
        prize_fill = races_df['PrizeMoney'].notna().sum()
        # Check for empty strings in PrizeMoney
        prize_fill = races_df[races_df['PrizeMoney'] != ''].shape[0]
            
        race_ids = tuple(races_df['RaceID'].tolist())
        if len(race_ids) == 1: race_ids = f"({race_ids[0]})"
        
        # Get entries stats
        query = f"""
        SELECT 
            COUNT(*) as Total,
            COUNT(CASE WHEN InRun IS NOT NULL AND InRun != '' THEN 1 END) as InRun_Fill,
            COUNT(CASE WHEN Comment IS NOT NULL AND Comment != '' THEN 1 END) as Comment_Fill,
            COUNT(CASE WHEN StartingPrice IS NOT NULL AND StartingPrice != '' THEN 1 END) as SP_Fill
        FROM GreyhoundEntries 
        WHERE RaceID IN {race_ids}
        """
        stats = pd.read_sql_query(query, conn)
        
        # Combine
        result = stats.iloc[0].to_dict()
        result['PrizeMoney_Fill'] = prize_fill
        result['Race_Count'] = len(races_df)
        
        print(f"DEBUG: Stats for {date}: Total={result['Total']}")
        return result, len(races_df)
    except Exception as e:
        print(f"DEBUG: Error in get_fill_rates: {e}")
        return None, 0

def verify_comparison():
    conn = sqlite3.connect(DB_PATH)
    
    print(f"Analyzing Target Date: {TARGET_DATE}...")
    target_stats, target_races = get_fill_rates(conn, TARGET_DATE)
    
    if target_stats is None or target_races == 0:
        print(f"❌ No data found for {TARGET_DATE}!")
        conn.close()
        return

    # Find a baseline date (first date in Dec before target with > 100 entries)
    print("Finding baseline date (early Dec)...")
    baseline_query = f"""
        SELECT MeetingDate, COUNT(*) as Count 
        FROM RaceMeetings 
        WHERE MeetingDate < '{TARGET_DATE}' AND MeetingDate >= '2025-12-01'
        GROUP BY MeetingDate 
        ORDER BY MeetingDate DESC
    """
    candidates = pd.read_sql_query(baseline_query, conn)
    
    baseline_date = None
    baseline_stats = None
    
    for date in candidates['MeetingDate']:
        stats, races = get_fill_rates(conn, date)
        if stats['Total'] > 100:
            baseline_date = date
            baseline_stats = stats
            break
            
    if not baseline_date:
        print("⚠️ Could not find a good baseline date in Dec. Checking recent history...")
        # Fallback to any recent
        baseline_date = '2025-12-01' # Force try
        baseline_stats, _ = get_fill_rates(conn, baseline_date)

    if baseline_stats is None:
        print("❌ Cannot compare - no baseline data found.")
        conn.close()
        return

    with open("verify_report.txt", "w", encoding="utf-8") as f:
        f.write(f"COMPARISON REPORT (EXTENDED)\n")
        f.write(f"{'Metric':<15} | {'Baseline (' + baseline_date + ')':<25} | {'Target (' + TARGET_DATE + ')':<25} | {'Status'}\n")
        f.write("-" * 85 + "\n")
        
        metrics = [
            ('PrizeMoney_Fill', 'Race_Count'), 
            ('InRun_Fill', 'Total'), 
            ('Comment_Fill', 'Total'), 
            ('SP_Fill', 'Total')
        ]
        
        for m, denom_key in metrics:
            base_denom = baseline_stats[denom_key]
            target_denom = target_stats[denom_key]
            
            base_count = baseline_stats[m]
            target_count = target_stats[m]
            
            base_pct = (base_count / base_denom) * 100 if base_denom > 0 else 0
            target_pct = (target_count / target_denom) * 100 if target_denom > 0 else 0
            
            diff = target_pct - base_pct
            
            status = "✅ OK"
            if target_pct < 50: status = "⚠️ LOW"
            if diff < -20: status = "❌ DROP"
            
            if m == 'PrizeMoney_Fill': m = 'PrizeMoney' 
            if m == 'InRun_Fill': m = 'Jump/InRun'
            if m == 'Comment_Fill': m = 'Comment'
            if m == 'SP_Fill': m = 'StartPrice'
            
            f.write(f"{m:<15} | {base_pct:.1f}% ({base_count}/{base_denom}){'':<5} | {target_pct:.1f}% ({target_count}/{target_denom}){'':<5} | {status}\n")

        f.write("-" * 85 + "\n")
        if target_stats['Total'] < 100:
            f.write("⚠️ Warning: Target date has low total volume.\n")
        else:
            f.write(f"Volume check: {target_stats['Total']} entries processed.\n")

    conn.close()

if __name__ == "__main__":
    verify_comparison()
