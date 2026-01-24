import os
import sys
from datetime import datetime

# Add root directory to path to allow importing src
sys.path.append(os.getcwd())
try:
    from src.integration.topaz_api import TopazAPI
    from src.core.config import TOPAZ_API_KEY
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.integration.topaz_api import TopazAPI
    from src.core.config import TOPAZ_API_KEY

def debug_topaz_runs():
    api = TopazAPI(TOPAZ_API_KEY)
    now = datetime.now()
    print(f"Fetching Topaz runs for {now.year}-{now.month}-{now.day} in VIC...")
    
    runs = api.get_bulk_runs_by_day('VIC', now.year, now.month, now.day)
    print(f"Found {len(runs)} runs.")
    if runs:
        # Check if Sale R6 is here
        print("Sample run keys:", runs[0].keys())
        sale_runs = [r for r in runs if 'SALE' in str(r.get('trackName')).upper() and r.get('raceNumber') == 6]
        print(f"Found {len(sale_runs)} runs for Sale R6.")
        for r in sale_runs:
            print(f"  {r.get('greyhoundName')} - Box: {r.get('boxNumber')} - Time: {r.get('raceTime')}")

if __name__ == "__main__":
    debug_topaz_runs()
