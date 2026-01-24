import requests

ssoid = "TjFRYP7wxAoeuLZud5Jp5bw3HtlnRuM5gRn55ZqApI4="
# cookies = {
#     "ssoid": ssoid
# }
headers = {
    "X-Authentication": ssoid,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json"
}

url = "https://identitysso.betfair.com/api/keepAlive"

print(f"Testing KeepAlive to {url} with X-Authentication...")
try:
    # Try GET first
    resp = requests.get(url, headers=headers) # No cookies
    print(f"GET Status: {resp.status_code}")
    print(f"GET Content: {resp.text[:500]}")
    
    if resp.status_code != 200:
        print("\nRetrying with POST...")
        resp = requests.post(url, cookies=cookies, headers=headers)
        print(f"POST Status: {resp.status_code}")
        print(f"POST Content: {resp.text[:500]}")

except Exception as e:
    print(f"Error: {e}")
