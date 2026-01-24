import requests

ssoid = "TjFRYP7wxAoeuLZud5Jp5bw3HtlnRuM5gRn55ZqApI4="

headers = {
    "X-Authentication": ssoid,
    "ssoid": ssoid, # Try both header and cookie equivalent
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

cookies = {
    "ssoid": ssoid
}

url = "https://historicdata.betfair.com/api/GetCollectionOptions"

print(f"Testing Historic Access to {url}...")
try:
    # Try with Header AND Cookie to be safe
    # Empty payload to see all available metadata
    resp = requests.post(url, headers=headers, cookies=cookies, json={
        "FromDay": 1,
        "FromMonth": 12,
        "FromYear": 2025,
        "ToDay": 31,
        "ToMonth": 12,
        "ToYear": 2025
    })
    print(f"Status: {resp.status_code}")
    # print(f"Content: {resp.text[:500]}")
    import json
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
