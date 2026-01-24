import requests

ssoid = "9V+k5Mgb9N+8L4IK3BCs1tkFciTA/uUIPFQGT7aPDw="
cookies = {
    "ssoid": ssoid
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Content-Type": "application/json"
}

url = "https://historicdata.betfair.com/api/GetCollectionOptions"
print(f"Testing Auth to {url} with cookies...")
try:
    resp = requests.post(url, cookies=cookies, headers=headers, json={})
    print(f"Status: {resp.status_code}")
    print(f"Headers: {resp.headers}")
    print(f"Content: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
