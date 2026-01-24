import requests
import betfair_data as bfd
import os

SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0="

def debug():
    # Reuse downloaded file if exists, else download
    # Assume file was downloaded previously or I download again? 
    # I deleted it in previous script (os.remove).
    # Need to download again.
    
    url = "https://historicdata.betfair.com/api/DownloadListOfFiles"
    payload = {
        "sport": "Greyhound Racing", "plan": "Basic Plan",
        "fromDay": 1, "fromMonth": 11, "fromYear": 2025,
        "toDay": 1, "toMonth": 11, "toYear": 2025,
        "marketTypesCollection": ["WIN"], "countriesCollection": ["AU"], "fileTypeCollection": ["M"]
    }
    headers = {'content-type': 'application/json', 'ssoid': SSOID}
    resp = requests.post(url, headers=headers, json=payload)
    files = resp.json()
    fpath = files[0]
    local = "temp_debug.bz2"
    
    from urllib.parse import quote
    encoded = quote(fpath, safe='')
    durl = f"https://historicdata.betfair.com/api/DownloadFile?filePath={encoded}"
    r = requests.get(durl, headers={'ssoid':SSOID})
    with open(local, 'wb') as f:
        f.write(r.content)
        
    print("Inspecting...")
    for f in bfd.Files([local]):
        for market in f:
            print(f"Venue: {getattr(market, 'venue', 'MISSING')}")
            print(f"MarketTime: {getattr(market, 'market_time', 'MISSING')}")
            runners = getattr(market, 'runners', 'MISSING')
            print(f"Runners Type: {type(runners)}")
            if runners != 'MISSING' and len(runners) > 0:
                print(f"Runner 0 Name: {runners[0].name}")
                
            count = 0
            for update in market:
                count += 1
            print(f"Updates Count: {count}")
            break
        break
    
    os.remove(local)

if __name__ == "__main__":
    debug()
