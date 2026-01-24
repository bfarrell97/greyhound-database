"""Check what track names BSP uses"""
import requests
import json
import bz2
from urllib.parse import quote
from datetime import datetime

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'

# Get files for November
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": 1, "fromMonth": 11, "fromYear": 2025,
    "toDay": 30, "toMonth": 11, "toYear": 2025,
    "eventId": None, "eventName": None,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}
headers = {'content-type': 'application/json', 'ssoid': SSOID}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
files = resp.json() if resp.status_code == 200 else []

print(f"Total files for Nov 2025: {len(files)}")

# Extract all unique venue names
venues = set()
for fp in files[:500]:  # Check first 500 files
    encoded_path = quote(fp, safe='')
    url = f"{BASE_URL}DownloadFile?filePath={encoded_path}"
    resp = requests.get(url, headers={'ssoid': SSOID}, timeout=30)
    if resp.status_code != 200:
        continue
    try:
        decompressed = bz2.decompress(resp.content).decode('utf-8')
        lines = decompressed.strip().split('\n')
        data = json.loads(lines[-1])
        if 'mc' in data and data['mc'] and 'marketDefinition' in data['mc'][0]:
            venue = data['mc'][0]['marketDefinition'].get('venue', '')
            if venue:
                venues.add(venue)
    except:
        pass

print(f"\nAll BSP venue names found:")
for v in sorted(venues):
    print(f"  {v}")
