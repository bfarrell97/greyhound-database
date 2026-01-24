
import sys
import os
sys.path.append(os.getcwd())
from src.integration.topaz_api import TopazAPI
from src.core.config import TOPAZ_API_KEY
import json

def debug_api():
    api = TopazAPI(TOPAZ_API_KEY)
    
    # Case 1: Gawler 16/12/2025
    print("\n--- DEBUGGING GAWLER 16/12/2025 ---")
    runs = api.get_bulk_runs_by_day('SA', 2025, 12, 16)
    
    found_gawler = False
    for run in runs:
        if run.get('trackCode') == 'GAW':
            found_gawler = True
            # Check for the specific dogs mentioned in validation failure
            # PRIME OF LIFE, SANDYCOVE in Race 2
            if run.get('raceNumber') == 2 and run.get('dogName') in ['PRIME OF LIFE', 'SANDYCOVE']:
                print(f"Race {run['raceNumber']} - {run['dogName']}")
                print(f"  ResultTime: {run.get('resultTime')}")
                print(f"  Place: {run.get('place')}")
                print(f"  Scratched: {run.get('scratched')}")
                print(f"  Status: {run.get('status')}") # if exists
                
    if not found_gawler:
        print("No Gawler runs found in SA bulk data.")

    # Case 2: Richmond Straight 18/12/2025
    print("\n--- DEBUGGING RICHMOND STRAIGHT 18/12/2025 ---")
    runs_nsw = api.get_bulk_runs_by_day('NSW', 2025, 12, 18)
    
    ris_count = 0
    null_time_count = 0
    
    for run in runs_nsw:
        # Richmond Straight might be RIC or RIS? Map says RIC: NSW. Check track name.
        if 'Richmond' in run.get('trackName', ''):
            ris_count += 1
            if run.get('resultTime') is None:
                null_time_count += 1
                if null_time_count <= 5: # Print first 5
                    print(f"Race {run['raceNumber']} - {run['dogName']} (Track: {run['trackName']})")
                    print(f"  ResultTime: None")
                    print(f"  Place: {run.get('place')}")
                    print(f"  Scratched: {run.get('scratched')}")

    print(f"\nTotal Richmond runs: {ris_count}")
    print(f"Runs with NULL resultTime: {null_time_count}")

if __name__ == "__main__":
    debug_api()
