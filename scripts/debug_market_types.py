import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.getcwd())

from src.integration.betfair_fetcher import BetfairOddsFetcher

def debug_markets():
    print("=== BETFAIR MARKET DIAGNOSTIC ===")
    fetcher = BetfairOddsFetcher()
    if fetcher.login():
        print("[OK] Logged in.")
        
        # Fetch next few hours
        now = datetime.utcnow()
        markets = fetcher.get_greyhound_markets(
            from_time=now - timedelta(minutes=30),
            to_time=now + timedelta(hours=2)
        )
        
        print(f"\nFetched {len(markets)} markets.")
        
        for m in markets:
            m_id = m.market_id
            m_name = m.market_name
            
            # Check Description
            desc_type = "UNKNOWN"
            if hasattr(m, 'description') and m.description:
                desc_type = getattr(m.description, 'market_type', 'MISSING')
            
            # Apply My Filter Logic
            is_win = True
            
            # 1. Metadata Check
            if desc_type != 'UNKNOWN' and desc_type != 'MISSING':
                if desc_type != 'WIN':
                    is_win = False
            
            # 2. Name Check
            name_lower = (m_name or '').lower()
            if 'place' in name_lower or 'tbp' in name_lower or 'forecast' in name_lower or 'quinella' in name_lower or ' 2 ' in name_lower or ' 3 ' in name_lower:
                is_win = False
                
            # SPECIAL DEBUG FOR "SWIFT CHOICE" (User mentioned this dog)
            # Check runners
            has_swift = False
            for r in m.runners:
                if 'SWIFT CHOICE' in r.runner_name.upper():
                    has_swift = True
                    break
            
            res_str = "WIN" if is_win else "SKIP"
            
            if has_swift or is_win: # Only print relevant
                print(f"[{res_str}] ID: {m_id} | Type: {desc_type:10} | Name: {m_name}")
                if has_swift:
                    print(f"   >>> CONTAINS SWIFT CHOICE")
                    
        fetcher.logout()
    else:
        print("[ERROR] Login failed.")

if __name__ == "__main__":
    debug_markets()
