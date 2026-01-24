import os
import requests
import json
from datetime import datetime

# Configuration
SSOID = "TjFRYP7wxAoeuLZud5Jp5bw3HtlnRuM5gRn55ZqApI4=" # Hardcoded verified SSOID
DOWNLOAD_DIR = "data/bsp"

# Mimic USER CURL headers EXACTLY
HEADERS = {
    "content-type": "application/json",
    "ssoid": SSOID
}
# Removed X-Authentication and custom User-Agent as they might be triggering blocks

def main():
    print("=== Betfair Historical Data Downloader (Custom Curl-Like) ===")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    url_list = "https://historicdata.betfair.com/api/GetFileList"
    
    # Try just the most standard combination first
    combo = {"sport": "Greyhound Racing", "plan": "Basic Plan"}
    print(f"Trying params: {combo}...")
    
    payload = {
        "fromDay": 1,
        "fromMonth": 11,
        "fromYear": 2025,
        "toDay": 30,
        "toMonth": 11,
        "toYear": 2025,
        "eventId": None,
        "eventName": None,
        "marketTypesCollection": [],
        "countriesCollection": [],
        "fileTypeCollection": [],
        **combo
    }
    
    files = [] # Initialize safety
    
    try:
        # Note: Sending json=payload automatically sets content-type usually, but we set it explicitly in HEADERS too.
        # Requests merges headers.
        
        resp = requests.post(url_list, headers=HEADERS, json=payload) # Removed cookies arg, relying on header
        
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            if resp.text.strip().startswith('['):
                files = resp.json()
                print("SUCCESS! File list retrieved.")
            else:
                 print(f"Got 200 OK but response is not a list (likely HTML error):")
                 print(resp.text[:500])
        else:
            print(f"Failed ({resp.status_code})")
            print(f"Response: {resp.text[:500]}")

    except Exception as e:
        print(f"Exc: {e}")
            
    if not files:
        print("Failed to retrieve file list.")
        return
        
    print(f"Found {len(files)} files.")
        
    # 2. Download Files
    for i, file_info in enumerate(files):
            filename = file_info.get("filename")
            download_url = file_info.get("downloadURL")
            
            if not filename or not download_url:
                continue
                
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            if os.path.exists(local_path):
                # print(f"[{i+1}/{len(files)}] Skipping {filename} (Exists)")
                continue
                
            print(f"[{i+1}/{len(files)}] Downloading {filename}...")
            
            try:
                # Use same headers for download
                r_file = requests.get(download_url, headers=HEADERS, stream=True)
                if r_file.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in r_file.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Failed to download {filename}: {r_file.status_code}")
            except Exception as e:
                print(f"Exception downloading {filename}: {e}")

if __name__ == "__main__":
    main()
