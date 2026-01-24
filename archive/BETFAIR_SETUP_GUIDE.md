# Betfair API Setup Guide

## Current Status
Your Betfair credentials are configured in `config.py`, but the API connection is returning a 403 Forbidden error. This typically means:

1. The App Key needs to be activated
2. The account needs API access enabled
3. Additional authentication steps are required

## Steps to Enable Betfair API Access

### 1. Verify Betfair Account
- Log in to https://www.betfair.com.au/
- Ensure your account is fully verified and funded

### 2. Apply for API Access
1. Go to https://developer.betfair.com/
2. Log in with your Betfair credentials
3. Navigate to "My Account" > "My API"
4. Click "Get a Free API App Key" if you haven't already
5. Fill out the application form:
   - Application Name: "Greyhound ML Predictions"
   - Description: "Personal use for greyhound racing predictions"
   - Purpose: "Personal betting automation"

### 3. Activate Your App Key
- Once approved, you'll receive an **Application Key (App Key)**
- This can take 24-48 hours to be approved
- Update `config.py` with your approved app key

### 4. Enable Non-Interactive Login
Betfair offers two login methods:

**Option A: Interactive Login (Current Method)**
- Uses username/password
- May require additional verification

**Option B: Certificate-Based Login (More Reliable)**
- Requires generating SSL certificates
- More secure and reliable for automated systems
- See: https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Non-Interactive+login

### Current Configuration
```python
BETFAIR_APP_KEY = "kd4sXjBoYYTS02z3"
BETFAIR_USERNAME = "bfarrell97@hotmail.com"
BETFAIR_PASSWORD = "Bradams900!"
```

## Testing Betfair Connection

Run this to test your Betfair API connection:
```bash
python betfair_api.py
```

Expected output (when working):
```
[OK] Logged in to Betfair successfully
[OK] Found X greyhound racing events
[OK] Found X markets
```

## Alternative: Manual Odds Entry

Until Betfair API is working, you have two options:

### Option 1: Use TAB/Other Bookmaker APIs
Some bookmakers offer public APIs that don't require as strict approval:
- TAB API (if available)
- Odds aggregator services

### Option 2: Scrape Odds from Website
- Use web scraping to get odds from Betfair or TAB website
- Less reliable but doesn't require API approval
- Would need to handle rate limiting and potential blocking

### Option 3: Manual Odds Import
- Export odds from a betting site to CSV
- Import into the database manually
- Good for testing the ML model while waiting for API approval

## Next Steps

1. **Wait for Betfair API approval** (if you've applied)
   - Check your email for approval notification
   - Verify app key is active in developer portal

2. **Test the connection** once approved:
   ```bash
   python betfair_api.py
   ```

3. **If still not working**, consider:
   - Contacting Betfair API support
   - Using certificate-based authentication
   - Alternative odds sources

## Using the Upcoming Betting Scraper Without Betfair

The scraper will still work and fetch race information from Topaz API. It will store:
- Race details (track, distance, time)
- Greyhound names
- Box numbers
- Trainer names
- Form data

The `CurrentOdds` field will be `NULL` until Betfair API is working.

You can still use the ML model to generate predictions - you'll just need to manually check odds on Betfair website to find value bets.
