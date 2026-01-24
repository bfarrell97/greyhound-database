import sys
import os
import json
sys.path.append(os.getcwd())
from src.integration.topaz_api import TopazAPI
try:
    from src.core.config import TOPAZ_API_KEY
except ImportError:
    # Fallback to key found in file if config missing
    TOPAZ_API_KEY = "313c5027-4e3b-4f5b-a1b4-3608153dbaa3"

def inspect():
    api = TopazAPI(TOPAZ_API_KEY)
    print("Fetching bulk runs for yesterday (VIC)...")
    # Use a recent date (e.g. yesterday) to ensure data exists
    # Assuming current date is 2025-12-24, let's try 2025-12-23
    data = api.get_bulk_runs_by_day('VIC', 2025, 12, 23)
    
    if not data:
        print("No data found for 2025-12-23. Trying 2025-01-01 (historical)...")
        data = api.get_bulk_runs_by_day('VIC', 2025, 1, 1)
        
    if not data:
        print("Still no data.")
        return

    print(f"Found {len(data)} runs.")
    print(f"Found {len(data)} runs.")
    if not data: return

    # Clean check
    keys = list(data[0].keys())
    time_keys = [k for k in keys if 'time' in k.lower()]
    split_keys = [k for k in keys if 'split' in k.lower()]
    sect_keys = [k for k in keys if 'sect' in k.lower()]
    
    print("Time Keys:", time_keys)
    print("Split Keys:", split_keys)
    print("Sector Keys:", sect_keys)
    print("PIR Key:", 'pir' in keys)
    print("Comment Key:", 'comment' in keys)
    
    # Check values for first run
    print("\nValues for First Run:")
    for k in time_keys + split_keys + ['pir', 'comment']:
        if k in data[0]:
            print(f"{k}: {data[0][k]}")

if __name__ == "__main__":
    inspect()
