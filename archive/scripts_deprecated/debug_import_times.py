
import logging
import os
import requests
import bz2
import shutil
import betfair_data as bfd
from datetime import datetime
import time

# Params
SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0=" 
DATA_DIR = "temp_debug_data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_sample():
    # Helper to download ONE file using pipeline logic
    year, month = 2025, 11
    url = "https://historicdata.betfair.com/api/DownloadListOfFiles"
    payload = {
        "sport": "Greyhound Racing",
        "plan": "Basic Plan",
        "fromDay": 1, "fromMonth": 11, "fromYear": 2025,
        "toDay": 1, "toMonth": 11, "toYear": 2025,
        "marketTypesCollection": ["WIN"],
        "countriesCollection": ["AU"],
        "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    
    print("Listing files...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        files = resp.json()
    except Exception as e:
        print(f"Error: {e}")
        return []

    if not files:
        print("No files found.")
        return []
        
    f_path = files[0] # String path
    print(f"Downloading {f_path}...")
    
    encoded_path = f_path.replace('/', '%2F') # Simple encode? No, rely on quote
    # Pipeline used requests.get with params? No, quote(path)
    from urllib.parse import quote
    encoded_path = quote(f_path, safe='')
    durl = f"https://historicdata.betfair.com/api/DownloadFile?filePath={encoded_path}"
    
    # Download
    r = requests.get(durl, headers={'ssoid': SSOID}, stream=True)
    out_path = os.path.join(DATA_DIR, os.path.basename(f_path))
    with open(out_path, 'wb') as f:
        f.write(r.content)
        
    return [out_path]

def inspect_times(path):
    print(f"Inspecting {path}...")
    
    # Use betfair_data
    file_obj = bfd.Files([path])
    
    for f in file_obj:
        market_time = None
        
        print(f"\n--- MARKET ---")
        updates = 0
        
        for update in f:
            updates += 1
            if hasattr(update, 'market_time') and update.market_time:
                market_time = update.market_time
                print(f"Market Time Found: {market_time}")
            
            if not market_time:
                continue
                
            if hasattr(update, 'publish_time'):
                pt = update.publish_time
                seconds_out = (market_time - pt).total_seconds()
                
                # Check for Price Data
                has_ltp = False
                for runner in update.runners:
                    if getattr(runner, 'ltp', None):
                        has_ltp = True
                        break
                
                if has_ltp and updates % 10 == 0: # Sample
                    print(f"Update {updates}: PT={pt}, SecsOut={seconds_out:.1f}s ({seconds_out/60:.1f}m)")
                    
                    if seconds_out >= 3500 and seconds_out <= 3700:
                        print(f"  *** TARGET ZONE T-60m DETECTED ***")
                        
        print(f"Total Updates: {updates}")

if __name__ == "__main__":
    paths = download_sample()
    if paths:
        inspect_times(paths[0])
