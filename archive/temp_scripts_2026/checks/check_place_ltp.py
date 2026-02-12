"""Check if PLACE markets have LTP data"""
import requests
import bz2
import json
from urllib.parse import quote

SSOID = 'Go3cWrYE2AaXgxNdU0Zs/HMWFmdAV9FAiPnI3wLJI6k='
BASE_URL = 'https://historicdata.betfair.com/api/'

# Get PLACE market file
url = BASE_URL + 'DownloadListOfFiles'
payload = {
    'sport': 'Greyhound Racing', 'plan': 'Basic Plan',
    'fromDay': 1, 'fromMonth': 11, 'fromYear': 2024,
    'toDay': 2, 'toMonth': 11, 'toYear': 2024,
    'marketTypesCollection': ['PLACE'], 'countriesCollection': ['AU'], 'fileTypeCollection': ['M']
}
resp = requests.post(url, headers={'content-type': 'application/json', 'ssoid': SSOID}, json=payload, timeout=30)
files = resp.json()
print(f'Got {len(files)} PLACE files')

if files:
    fp = files[0]
    encoded = quote(fp, safe='')
    resp = requests.get(f'{BASE_URL}DownloadFile?filePath={encoded}', headers={'ssoid': SSOID}, timeout=30)
    data = bz2.decompress(resp.content).decode('utf-8')
    lines = data.strip().split('\n')
    last = json.loads(lines[-1])
    md = last['mc'][0]['marketDefinition']
    
    mtype = md.get('marketType')
    venue = md.get('venue')
    print(f'Market type: {mtype}')
    print(f'Venue: {venue}')
    
    # Check for LTP data
    has_ltp = False
    for line in lines[10:50]:
        try:
            d = json.loads(line)
            if 'mc' in d and d['mc'] and d['mc'][0].get('rc'):
                rc = d['mc'][0]['rc'][0]
                if 'ltp' in rc:
                    has_ltp = True
                    print(f'Sample rc: {rc}')
                    break
        except:
            continue
    print(f'\nHas LTP data: {has_ltp}')
