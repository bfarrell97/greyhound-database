# Multi-Race Scraping Feature

## Overview

The scraper can now automatically scrape **ALL races** from a meeting with a single click. Previously, you had to scrape each race individually by entering 12 different URLs. Now, you can scrape all 12 races from one URL.

## How It Works

When you provide a results URL like:
```
https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/
```

The scraper will:
1. Load the page
2. Find all race selector buttons (1, 2, 3, ..., 12)
3. Click each button sequentially
4. Scrape the data after each click
5. Save all races to the database

## Using the GUI

1. Open the GUI: `python greyhound_racing_gui.py`
2. Go to "Scrape Data" tab
3. Paste a results URL
4. Enter the race date
5. Make sure **"Scrape entire meeting (all races)"** is checked ✓
6. Click "Scrape and Save to Database"

The browser will open and you'll see it clicking through each race automatically.

## Using the Test Script

To test the feature:
```bash
python test_multi_race_scraper.py
```

This will scrape all 12 races from the Ballarat meeting on Nov 29, 2025 and save them to the database.

## Technical Details

### Selector Discovery

The scraper tries multiple CSS selectors to find the race buttons:
- `.meeting-events-nav__item` (current structure)
- `button.meeting-events-nav__item`
- `.meeting-race-number-selector__button`
- `button.meeting-race-number-selector__button`

This ensures compatibility if the website changes its structure.

### Performance

- **Wait time after clicking**: 1.5 seconds (user reported fast loading)
- **Initial page load**: 5 seconds (to allow JavaScript to render)
- **Total time for 12 races**: Approximately 2-3 minutes

### Code Changes

**Files Modified:**
1. [greyhound_scraper_v2.py](greyhound_scraper_v2.py) - Added `scrape_all_meeting_results()` method (lines 461-681)
2. [greyhound_racing_gui.py](greyhound_racing_gui.py) - Added checkbox and multi-race logic
3. [QUICK_START.md](QUICK_START.md) - Updated instructions

**Files Created:**
1. [test_multi_race_scraper.py](test_multi_race_scraper.py) - Test script for multi-race scraping

### Key Features

✅ Automatically finds all races in a meeting
✅ Handles meetings with different numbers of races (8, 10, 12, etc.)
✅ Saves each race individually to the database
✅ Shows progress in the GUI log
✅ Fallback to single race if buttons not found
✅ Works with non-headless mode for reliability

## Limitations

- **Results URLs only**: Multi-race scraping only works for completed race results
- **Form guides**: Each form guide must still be scraped individually (different race IDs)
- **Single meeting**: Can only scrape one meeting at a time (use batch scraper for multiple meetings)

## Example Output

```
================================================================================
TESTING MULTI-RACE SCRAPER
================================================================================

Found 12 races using selector: .meeting-events-nav__item
Found 12 races in this meeting

[1/12] Scraping Race 1...
  Found 8 results for Race 1

[2/12] Scraping Race 2...
  Found 10 results for Race 2

...

[12/12] Scraping Race 12...
  Found 10 results for Race 12

================================================================================
DATABASE SAVE COMPLETE
================================================================================
Success: 12/12
Failed: 0/12
================================================================================
```

## Troubleshooting

### "No race selector buttons found"

This usually means:
- The page is still loading (wait longer)
- It's a form guide URL (multi-race only works for results)
- The website changed its structure

### Some races fail to save

- Check the log output to see which races failed
- May be due to duplicate data (already in database)
- Try scraping those races individually

## Future Enhancements

- Add multi-meeting scraping (scrape all meetings for a track/date)
- Add progress bar showing which race is being scraped
- Add option to skip races already in database
- Support form guide multi-race scraping
