import requests
import json

SSOID = "TjFRYP7wxAoeuLZud5Jp5bw3HtlnRuM5gRn55ZqApI4="

# url = "https://historicdata.betfair.com/api/GetCollectionOptions"
# url = "https://historicdata.betfair.com/api/GetFileList"
url = "https://historicdata.betfair.com/api/GetMyData"

# Mimic USER CURL exactly
headers = {
    'content-type': 'application/json',
    'ssoid': SSOID
}

# GetMyData usually requires empty payload or no payload
payload = {}

print("--- Sending Request (Mimic Curl) ---")
print(f"URL: {url}")
print(f"Headers: {headers}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    resp = requests.post(url, headers=headers, json=payload)
    print(f"\nStatus: {resp.status_code}")
    print(f"Response Start: {resp.text[:500]}")
    
    if resp.status_code == 200 and resp.text.strip().startswith('{'):
        try:
            print("Response JSON Keys:", resp.json().keys())
        except:
            pass
except Exception as e:
    print(f"Error: {e}")
