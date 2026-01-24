# Greyhound Scraper Usage Guide

## Quick Start

1. Launch the GUI: `python greyhound_racing_gui.py`
2. Go to the "Scrape Data" tab
3. Enter a URL from The Greyhound Recorder website
4. Set the race date
5. Click "Scrape and Save to Database"

## Finding URLs to Scrape

### Results Pages (Completed Races)

1. Go to: https://www.thegreyhoundrecorder.com.au/results/
2. Select a track and date
3. Click on a race to view results
4. Copy the URL (format: `.../results/TRACK/RACE_ID/`)

**Example Results URL:**
```
https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/
```

### Form Guide Pages (Upcoming Races)

1. Go to: https://www.thegreyhoundrecorder.com.au/form-guides/
2. Select a track and date
3. Click "Long Form" for a race
4. Copy the URL (format: `.../form-guides/TRACK/long-form/RACE_ID/1/`)

**Example Form Guide URL:**
```
https://www.thegreyhoundrecorder.com.au/form-guides/broken-hill/long-form/248580/1/
```

## What Data Gets Scraped?

### From Results Pages:
- Race information (track, distance, grade, prize money)
- Greyhound finishing positions
- Finish times and margins
- Box numbers
- Trainers
- Split times (first section)
- In-run positions
- Weights
- Sire and Dam (breeding)
- Starting prices

### From Form Guides:
- Upcoming race information
- Greyhound entries and box numbers
- Trainers and owners
- Greyhound statistics (starts, wins, prizemoney)
- Track records (wins-runs-places at the track)
- Track/Distance records (wins-runs-places at specific track and distance)
- Breeding information (Sire x Dam)

## Troubleshooting

### "ERROR - No data scraped"

**Possible causes:**
1. **Website blocking**: The website uses CloudFlare protection which may block automated access
2. **Incorrect URL**: Make sure you're using a valid results or form guide URL
3. **Page still loading**: The browser window needs time to load JavaScript content

**Solutions:**
- The scraper is set to use a visible browser window (headless=False) which is more reliable
- Wait for the browser window to fully load before the scraper captures the data
- Try the URL in a regular browser first to make sure it's valid
- If you see "ERROR: The request could not be satisfied" in the page title, the website is blocking you

### Browser Window Opens But Nothing Happens

- This is normal - the scraper needs time to wait for JavaScript to render
- The browser will automatically close when scraping is complete
- Check the log output in the GUI for progress

### Date Format

- Use YYYY-MM-DD format (e.g., 2025-11-30)
- For results pages, use the date the race occurred
- For form guides, use the scheduled race date

## Testing the Scraper

You can test the scraper without the GUI:

### Test Results Scraper:
```bash
python test_full_results_scraper.py
```

### Test Form Guide Scraper:
```bash
python test_extract_data.py
```

### Test Complete Workflow (Scrape → Database):
```bash
python test_full_workflow.py
```

## Anti-Detection Features

The scraper includes several features to avoid being blocked:

1. **User-Agent Spoofing**: Pretends to be a regular Chrome browser
2. **Automation Detection Removal**: Hides the fact that it's controlled by Selenium
3. **New Headless Mode**: Uses Chrome's newer headless implementation
4. **JavaScript Execution**: Removes the `navigator.webdriver` property

These features work in both headless and non-headless modes.

## Database Storage

All scraped data is automatically saved to `greyhound_racing.db` with the following structure:

- **Greyhounds** - Name, sire, dam, statistics
- **Trainers** - Trainer information
- **Owners** - Owner information
- **Tracks** - Track names and locations
- **RaceMeetings** - Race meeting dates
- **Races** - Individual race information
- **GreyhoundEntries** - Greyhound entries in races with results
- **Benchmarks** - Track/distance benchmark times

## Tips for Best Results

1. **Start with Results**: Scrape historical results first to build up your database
2. **One Race at a Time**: Scrape one URL at a time for reliability
3. **Check the Log**: Always review the log output to confirm data was saved
4. **Verify in Database Viewer**: Use the "Database Viewer" tab to confirm data was saved correctly
5. **Consistent Dates**: Make sure the date you enter matches the race date

## Known Limitations

- The scraper can only access one race at a time (no batch scraping yet)
- Some older races may have different HTML structure
- CloudFlare protection may occasionally block access
- Historical form data is NOT scraped from form guides (only upcoming race info)

## Need Help?

Check the log output in the GUI for detailed error messages. Most issues are related to:
- Incorrect URL format
- Website blocking automated access
- Network connectivity issues
