import sqlite3
import pandas as pd
import numpy as np

def audit_leakage():
    print("AUDITING STRATEGY FOR DATA LEAKAGE")
    print("=" * 60)
    
    # 1. Load Bets
    csv_file = 'results/backtest_longterm_top3.csv'
    try:
        bets = pd.read_csv(csv_file)
        print(f"Loaded {len(bets)} bets from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Add 'RaceKey' for join if needed, but we have GreyhoundID + MeetingDate
    # We need to query DB for ACTUAL Split and Prize for these races
    
    conn = sqlite3.connect('greyhound_racing.db')
    
    print("Fetching actual race results from DB for comparison...")
    
    # Get unique dogs in bets
    dog_ids = bets['GreyhoundID'].unique()
    dogs_str = ",".join(map(str, dog_ids))
    
    # Query all results for these dogs (to map back to the specific races)
    # We match on GreyhoundID and MeetingDate (Approx unique key)
    # Note: Using MeetingDate is safe enough for 99.9% cases
    
    query = f"""
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        ge.Split as ActualSplit,
        ge.PrizeMoney as ActualRacePrize
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.GreyhoundID IN ({dogs_str})
    """
    
    try:
        actuals = pd.read_sql_query(query, conn)
        actuals['MeetingDate'] = pd.to_datetime(actuals['MeetingDate'])
        bets['MeetingDate'] = pd.to_datetime(bets['MeetingDate'])
        
        # Merge
        merged = bets.merge(actuals, on=['GreyhoundID', 'MeetingDate'], how='left')
        
        print(f"Merged {len(merged)} records. Verifying...")
        
        # 3. RunningPrize Integrity Check
        # RunningPrize should be cumulative SUM of PAST prizes.
        # It should NOT correlate 1.0 with ActualRacePrize.
        # Actually, it shouldn't correlate much at all with the CURRENT prize.
        
        if 'RunningPrize' not in merged.columns:
            print("WARNING: 'RunningPrize' column missing in CSV. Cannot audit Prize Leakage directly.")
        else:
            merged['ActualRacePrize'] = merged['ActualRacePrize'].fillna(0)
            corr_prize = merged['RunningPrize'].corr(merged['ActualRacePrize'])
            print(f"Correlation (RunningPrize vs ActualRacePrize): {corr_prize:.4f}")
            
            if corr_prize > 0.8:
                print(">>> CRITICAL LEAK: RunningPrize highly correlated with Current Race Prize!")
            else:
                print("PASS: RunningPrize does not predict Current Prize (Correlation low).")

        # 4. HistAvgSplit Integrity Check
        # HistAvgSplit should not predict ActualSplit perfectly
        if 'HistAvgSplit' not in merged.columns:
             print("WARNING: 'HistAvgSplit' missing.")
        else:
            # Drop NaNs
            valid = merged.dropna(subset=['HistAvgSplit', 'ActualSplit'])
            corr_split = valid['HistAvgSplit'].corr(valid['ActualSplit'])
            print(f"Correlation (HistAvgSplit vs ActualSplit): {corr_split:.4f}")
            
            if corr_split > 0.9:
                print(">>> CRITICAL LEAK: HistAvgSplit predicts ActualSplit too well (>0.9)!")
            elif corr_split > 0.6:
                print("INFO: HistAvgSplit correlates reasonably (0.6-0.9). This is expected for consistent dogs.")
            else:
                print("PASS: Split correlation is naturally low/moderate.")
            
            # Check for EXACT matches (Leakage)
            exact_matches = (valid['HistAvgSplit'] == valid['ActualSplit']).sum()
            print(f"Exact Matches (Hist == Actual): {exact_matches} / {len(valid)}")
            if exact_matches > len(valid) * 0.1:
                 print(">>> WARNING: significant number of EXACT matches. Suspicious.")

        # 5. Future Data Check (CareerPrizeMoney)
        # Check if 'CareerPrizeMoney' in CSV matches 'RunningPrize' 
        # (Assuming CSV has both, or we only have RunningPrize now)
        if 'CareerPrizeMoney' in bets.columns and 'RunningPrize' in bets.columns:
             print("Compare CSV CareerPrizeMoney vs RunningPrize...")
             # ...
             
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    audit_leakage()
