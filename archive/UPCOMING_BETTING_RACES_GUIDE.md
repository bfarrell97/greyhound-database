# Upcoming Betting Races Guide

## Overview

The upcoming betting races scraper fetches race cards for future greyhound races with current betting odds, allowing the ML model to make predictions on races that haven't been run yet.

## Files Created

1. **[upcoming_betting_scraper.py](upcoming_betting_scraper.py)** - Main scraper
   - Fetches upcoming races from Topaz API
   - Integrates with Betfair API for current odds
   - Stores data in database

2. **[betfair_odds_fetcher.py](betfair_odds_fetcher.py)** - Betfair API integration
   - Uses `betfairlightweight` library
   - Handles authentication and odds fetching
   - Maps odds to box numbers

3. **[betfair_api.py](betfair_api.py)** - Custom Betfair API client (alternative)
   - Direct API implementation
   - Currently has authentication issues
   - Use betfair_odds_fetcher.py instead

## Database Tables

The scraper creates two new tables:

### UpcomingBettingRaces
```sql
CREATE TABLE UpcomingBettingRaces (
    UpcomingBettingRaceID INTEGER PRIMARY KEY AUTOINCREMENT,
    MeetingDate TEXT NOT NULL,
    TrackCode TEXT NOT NULL,
    TrackName TEXT NOT NULL,
    RaceNumber INTEGER NOT NULL,
    RaceTime TEXT,
    Distance INTEGER,
    RaceType TEXT,
    LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(MeetingDate, TrackCode, RaceNumber)
)
```

### UpcomingBettingRunners
```sql
CREATE TABLE UpcomingBettingRunners (
    UpcomingBettingRunnerID INTEGER PRIMARY KEY AUTOINCREMENT,
    UpcomingBettingRaceID INTEGER NOT NULL,
    GreyhoundName TEXT NOT NULL,
    BoxNumber INTEGER,
    CurrentOdds REAL,
    TrainerName TEXT,
    Form TEXT,
    BestTime TEXT,
    Weight REAL,
    LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (UpcomingBettingRaceID) REFERENCES UpcomingBettingRaces(UpcomingBettingRaceID)
)
```

## Usage

### Scrape Tomorrow's Races
```bash
python upcoming_betting_scraper.py
```

### Scrape Next 7 Days
```python
from upcoming_betting_scraper import UpcomingBettingScraper

scraper = UpcomingBettingScraper()
scraper.scrape_next_n_days(7)
```

### Scrape Specific Date
```python
from upcoming_betting_scraper import UpcomingBettingScraper

scraper = UpcomingBettingScraper()
scraper.scrape_date('2025-12-10')
```

## Current Status

✅ **Working:**
- Topaz API integration - fetching upcoming races
- Race details (track, distance, race number, time)
- Greyhound names, box numbers, trainer names, form
- Database storage
- Betfair API login successful

⚠️ **Needs Attention:**
- Betfair API returning 0 markets
- Current odds field is NULL

## Betfair Issues

The Betfair API is successfully logging in but returning 0 markets. Possible causes:

1. **Account Not Funded**
   - Betfair may restrict API access to unfunded accounts
   - Try depositing funds into your Betfair account

2. **No Markets Available**
   - AU greyhound markets might not be available at the time of testing
   - Try running the scraper closer to race time

3. **Account Restrictions**
   - Your account might need additional verification
   - Contact Betfair support to enable API access

4. **Market Filters**
   - The search might be too restrictive
   - May need to adjust event type IDs or market filters

## Alternative Odds Sources

While waiting for Betfair to work, you can:

1. **Manual Odds Entry**
   - Export odds from Betfair website to CSV
   - Update database manually

2. **Web Scraping**
   - Scrape odds directly from Betfair/TAB websites
   - Less reliable, may violate terms of service

3. **Other Bookmaker APIs**
   - TAB API (if available)
   - Other bookmakers with public APIs

## Testing

Check what data was scraped:
```bash
python check_upcoming_betting.py
```

Test Betfair connection:
```bash
python betfair_odds_fetcher.py
```

## Next Steps

1. **Verify Betfair Account**
   - Check account is funded
   - Verify API access is enabled
   - Check for any account restrictions

2. **Test at Race Time**
   - Run scraper 30-60 minutes before first race
   - Markets typically open 24-48 hours before race

3. **Contact Betfair Support**
   - If still no markets, contact Betfair API support
   - Explain you're using the API for personal betting automation

4. **Integrate with ML Model**
   - Once odds are working, connect to prediction system
   - Filter for value bets (model prob > implied prob)
   - Apply 80% confidence threshold and $1.50 odds filter

## Data Scraped Successfully

Current test run scraped:
- **37 races** for 2025-12-09
- **282 runners** across Geelong, Warragul, and Horsham
- All race details and greyhound information stored
- Only missing: CurrentOdds (awaiting Betfair markets)

The infrastructure is ready - just needs Betfair markets to populate odds!
