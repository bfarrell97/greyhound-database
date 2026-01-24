import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
from betfairlightweight import filters

# Add src to path
sys.path.append(os.getcwd())
try:
    from src.integration.betfair_fetcher import BetfairOddsFetcher
except ImportError:
    print("Error: Could not import BetfairOddsFetcher")
    sys.exit(1)

def reconcile_bets():
    import os
    csv_path = 'live_bets.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean BetID
    df['BetID'] = df['BetID'].astype(str).str.replace('.0', '', regex=False)

    # Force 90 day lookback (UTC)
    from_date = (datetime.utcnow() - timedelta(days=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    print(f"Fetching settled orders since {from_date} (UTC)...")
    
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Failed to log in to Betfair.")
        return

    # RECONCILIATION VIA ACCOUNT STATEMENT
    # (Fallback since list_cleared_orders is returning 0)
    print("\n--- RECONCILING VIA ACCOUNT STATEMENT ---")
    
    try:
        # Fetch Statement with Pagination
        all_items = []
        start_record = 0
        batch_size = 100
        
        while True:
            print(f"Fetching statement batch from record {start_record}...")
            stmt = fetcher.trading.account.get_account_statement(
                item_date_range=filters.time_range(from_=from_date),
                from_record=start_record,
                record_count=batch_size
            )
            items = getattr(stmt, 'account_statement', [])
            if not items:
                break
                
            all_items.extend(items)
            print(f"  Got {len(items)} items. Total: {len(all_items)}")
            
            if stmt.more_available:
                start_record += len(items)
            else:
                break
            
            # Safety break
            if len(all_items) > 5000: break
            
        print(f"Fetched {len(all_items)} total statement transactions.")
        
        # Build Map: RefID -> Profit
        # Note: A single bet might have multiple entries (Stake, Commission, Payout)? 
        # Usually Statement aggregates per transaction Ref?
        # Let's map RefID -> Sum(Amount)
        
        profit_map = {}
        processed_refs = set()
        
        for i in all_items:
            ref = str(i.ref_id)
            amount = float(i.amount or 0.0)
            
            # Debug Print (First 20)
            if len(processed_refs) < 20: 
                 print(f"  Stmt: {i.item_date} | Ref: {ref} | Amt: {amount} | Class: {i.item_class}")
                 processed_refs.add(ref)
            
            if ref not in profit_map:
                profit_map[ref] = 0.0
            profit_map[ref] += amount

        # Update CSV
        updated_count = 0
        
        for idx, row in df.iterrows():
            raw_bet_id = str(row['BetID'])
            bet_id = raw_bet_id.replace('.0', '').strip()
            
            if bet_id in profit_map:
                real_profit = profit_map[bet_id]
                
                # Determine Result
                result = "WIN" if real_profit > 0 else "LOSS"
                if real_profit == 0: result = "VOID" # Or Break Even
                
                # Update DataFrame
                df.at[idx, 'Profit'] = real_profit
                df.at[idx, 'Status'] = 'SETTLED'
                df.at[idx, 'Result'] = result
                updated_count += 1
                print(f"MATCH: Bet {bet_id} -> ${real_profit:.2f} ({result})")
        
        print(f"\nUpdated {updated_count} records from Statement.")
        
        if updated_count > 0:
            df.to_csv(csv_path, index=False)
            print("Saved updated live_bets.csv")
            
            # Auto-Generate Report
            print("Generating P/L Summary Report...")
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from generate_pl_report import generate_report
                generate_report()
                print("P/L Summary Report Updated.")
            except Exception as report_e:
                 print(f"Error generating report: {report_e}")
                 # Fallback: Run as subprocess if import fails
                 try:
                     import subprocess
                     subprocess.run([sys.executable, 'scripts/generate_pl_report.py'], check=True)
                 except: pass
        else:
            print("No matches found between CSV BetIDs and Statement RefIDs.")
            
    except Exception as e:
        print(f"Reconciliation Error: {e}")
        import traceback
        traceback.print_exc()

    fetcher.logout()
    return

if __name__ == "__main__":
    reconcile_bets()
