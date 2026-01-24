import sys
import os
sys.path.append(os.getcwd())

from src.integration.topaz_api import TopazAPI
from src.core.config import TOPAZ_API_KEY
from datetime import datetime, timedelta
import json

def inspect_response():
    api = TopazAPI(TOPAZ_API_KEY)
    
    # Get runs for yesterday
    yesterday = datetime.now() - timedelta(days=1)
    year = yesterday.year
    month = yesterday.month
    day = yesterday.day
    
    print(f"Fetching bulk runs for {year}-{month}-{day} (VIC)...")
    try:
        runs = api.get_bulk_runs_by_day('VIC', year, month, day)
        if not runs:
            print("No runs found for yesterday. Trying 2 days ago...")
            yesterday = datetime.now() - timedelta(days=2)
            runs = api.get_bulk_runs_by_day('VIC', yesterday.year, yesterday.month, yesterday.day)
            
        if runs:
            print(f"Found {len(runs)} runs")
            run = runs[0]
            print("\nKeys in run object:")
            for k in sorted(run.keys()):
                print(f"  {k}: {type(run[k])} = {str(run[k])[:50]}")
                
            # Check for prize-like keys
            print("\nPrize related keys:")
            for k in run.keys():
                if 'prize' in k.lower() or 'money' in k.lower():
                    print(f"  {k}: {run[k]}")
        else:
            print("No runs found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_response()
