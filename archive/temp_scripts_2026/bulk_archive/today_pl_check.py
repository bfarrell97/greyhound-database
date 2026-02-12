
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add path to find 'src'
sys.path.append(os.getcwd())
try:
    from src.integration.betfair_fetcher import BetfairOddsFetcher
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.integration.betfair_fetcher import BetfairOddsFetcher

def calculate_today_pl():
    print(f"Connecting to Betfair to calculate Today's P/L ({datetime.now().strftime('%Y-%m-%d')})...")
    
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("Failed to login to Betfair.")
        return

    # Fetch Cleared Orders (Settled) for Last 24 Hours
    orders = fetcher.get_cleared_orders(days=1)
    fetcher.logout()
    
    if not orders:
        print("No settled orders found for today.")
        return

    print(f"Found {len(orders)} settled orders.")

    back_pl = 0.0
    lay_pl = 0.0
    back_count = 0
    lay_count = 0
    
    # Analyze
    # Note: 'orders' is a list of ClearedOrderSummary objects
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    today_date = datetime.now().date()
    print(f"Filtering for orders settled on or after {today_date} (Local Time)...")
    
    processed_orders = []
    
    for o in orders:
        # Handle Date Timezone (Betfair returns UTC)
        # We need to convert to Local Time to check if it matches "Today"
        # Assuming o.settled_date is a datetime object. If it's a string, verify.
        # list_cleared_orders usually returns strings or datetimes depending on the client library version. 
        # But let's handle both.
        
        s_date = o.settled_date
        if isinstance(s_date, str):
            # Parse ISO buffer 
            # e.g. 2025-12-31T09:00:00.000Z
            try:
                s_dt = datetime.strptime(s_date.replace('Z', '+0000'), '%Y-%m-%dT%H:%M:%S.%f%z')
            except:
                # Fallback format
                try:
                    s_dt = datetime.strptime(s_date, '%Y-%m-%d %H:%M:%S')
                except:
                    continue
        else:
            s_dt = s_date
            
        # Convert to local (System Time is +11:00 roughly)
        # Use astimezone() if tz-aware, else assume it's already relevant or adjust
        if s_dt.tzinfo:
            local_dt = s_dt.astimezone()
        else:
            # If naive, assume UTC and convert? Or assume local?
            # Betfair fetcher usually returns naive UTC if using lightweight.
            # Let's explicitly assume UTC if naive for safety.
            from datetime import timezone
            s_dt = s_dt.replace(tzinfo=timezone.utc)
            local_dt = s_dt.astimezone()
            
        if local_dt.date() != today_date:
            continue

        # P/L logic
        profit = o.profit if o.profit else 0.0
        side = o.side # 'BACK' or 'LAY'
        
        processed_orders.append({
            'Market': o.event_id, 
            'Selection': o.selection_id,
            'Side': side,
            'Profit': profit,
            'Settled': local_dt.strftime('%H:%M:%S')
        })

        if side == 'BACK':
            back_pl += profit
            back_count += 1
        elif side == 'LAY':
            lay_pl += profit
            lay_count += 1
            
    # Print Summary
    print("\n" + "="*40)
    print(f"BETFAIR P/L SUMMARY ({today_str})")
    print("="*40)
    
    print(f"BACK BETS: {back_count}")
    print(f"BACK P/L:  ${back_pl:+.2f}")
    print("-" * 20)
    
    print(f"LAY BETS:  {lay_count}")
    print(f"LAY P/L:   ${lay_pl:+.2f}")
    print("-" * 20)
    
    total_pl = back_pl + lay_pl
    print(f"TOTAL P/L: ${total_pl:+.2f}")
    print("="*40)

if __name__ == "__main__":
    calculate_today_pl()
