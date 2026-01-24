"""Explore Betfair Historical Data API"""
import requests
import json

SSOID = 'z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc='
BASE_URL = 'https://historicdata.betfair.com/api/'

headers = {'ssoid': SSOID}

print('Getting available data...', flush=True)
resp = requests.get(BASE_URL + 'GetMyData', headers=headers, timeout=30)
data = resp.json()

print(f'Found {len(data)} data items', flush=True)
print()

# Show structure
for item in data[:10]:
    print(f"Sport: {item.get('sport')}, Plan: {item.get('plan')}, Date: {item.get('forDate')[:10]}", flush=True)

print()
print("Unique sports:", set(item.get('sport') for item in data), flush=True)

# Check if there's a download URL
print()
print("Sample item full structure:", flush=True)
print(json.dumps(data[0], indent=2), flush=True)
