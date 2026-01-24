"""Test Betfair API connection"""
import requests

SSOID = "LJ2QtRWn4DrTzcb9ZgQdJc2H5McN723K0la2wEaDHH0="
url = "https://historicdata.betfair.com/api/DownloadListOfFiles"

payload = {
    "sport": "Greyhound Racing",
    "plan": "Basic Plan",
    "fromDay": 1, "fromMonth": 11, "fromYear": 2024,
    "toDay": 30, "toMonth": 11, "toYear": 2024,
    "eventId": None, "eventName": None,
    "marketTypesCollection": ["WIN"],
    "countriesCollection": ["AU"],
    "fileTypeCollection": ["M"]
}

headers = {'content-type': 'application/json', 'ssoid': SSOID}

print("Testing Betfair Historical Data API...")
print(f"URL: {url}")
print(f"SSOID: {SSOID[:20]}...")

try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Response length: {len(r.text)}")
    print(f"Raw response:\n{r.text}")
except Exception as e:
    print(f"Error: {e}")
