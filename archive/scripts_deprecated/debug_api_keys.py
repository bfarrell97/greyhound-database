"""Debug: Show exact API key format vs lookup key format"""
import sqlite3
import requests
import bz2
import json
import re
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'

def normalize_name(name):
    match = re.match(r'\d+\.\s*(.+)', name)
    if match:
        name = match.group(1)
    name = name.upper()
    name = re.sub(r"['\-\.]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

# Get one Capalaba file from API
dt = datetime(2022, 6, 5)
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": dt.day, "fromMonth": dt.month, "fromYear": dt.year,
    "toDay": dt.day, "toMonth": dt.month, "toYear": dt.year,
    "eventId": None, "eventName": None,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}
headers = {'content-type': 'application/json', 'ssoid': SSOID}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
files = resp.json() if resp.status_code == 200 else []

print("Checking API files...")
print(f"Total files: {len(files)}")

for fp in files[:100]:  # Check first 100
    encoded_path = quote(fp, safe='')
    furl = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    try:
        fresp = requests.get(furl, headers={'ssoid': SSOID}, timeout=30)
        if fresp.status_code != 200:
            continue
        decompressed = bz2.decompress(fresp.content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        data = json.loads(lines[-1])
        if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
            md = data['mc'][0]['marketDefinition']
            venue = md.get('venue', '').upper()
            market_time_str = md.get('marketTime', '')[:10]
            
            if 'CAPALABA' in venue:
                print(f"\nFOUND CAPALABA FILE!")
                print(f"  File: {fp}")
                print(f"  Venue: '{venue}'")
                print(f"  MarketTime: '{market_time_str}'")
                
                for r in md.get('runners', []):
                    name = normalize_name(r.get('name', ''))
                    bsp = r.get('bsp')
                    if name and bsp and bsp > 0:
                        key = (venue, market_time_str, name)
                        print(f"  API Key: {key}")
                        break  # Just show one
                break  # Just check one Capalaba file
    except:
        continue
