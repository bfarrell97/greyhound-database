# Quick Start Guide

## Installation (First Time Only)

1. Make sure Python 3.11+ and Chrome are installed
2. Open terminal/command prompt in this folder
3. Run: `pip install -r requirements.txt`

## Daily Usage

### Option 1: GUI (Easiest)

```bash
python greyhound_racing_gui.py
```

1. Click "Scrape Data" tab
2. Paste URL from https://www.thegreyhoundrecorder.com.au/
3. Enter race date (YYYY-MM-DD format)
4. Check "Scrape entire meeting" to get ALL races from the meeting (recommended for results URLs)
5. Click "Scrape and Save to Database"
6. Wait for browser to finish (it will close automatically)

### Option 2: Batch Scraper

For scraping multiple races:

```bash
python batch_scraper.py
```

Then choose:
- **Option 1**: Enter URLs one by one interactively
- **Option 2**: Read from a file (create urls.txt first)
- **Option 3**: Single URL from command line

### Option 3: Command Line

```bash
python batch_scraper.py "https://www.thegreyhoundrecorder.com.au/results/ballarat/248601/" "2025-11-29"
```

## Finding URLs

**Results (Completed Races):**
1. Go to https://www.thegreyhoundrecorder.com.au/results/
2. Select track and date
3. Click on a race
4. Copy the URL

**Form Guides (Upcoming Races):**
1. Go to https://www.thegreyhoundrecorder.com.au/form-guides/
2. Select track and date
3. Click "Long Form" on a race
4. Copy the URL

## Viewing Data

### Database Viewer
```bash
python view_database.py
```

### Database Stats
```bash
python check_database.py
```

### Clear Database
```bash
python clear_database.py
```

## Common URLs Format

**Results:**
```
https://www.thegreyhoundrecorder.com.au/results/TRACK_NAME/RACE_ID/
```

**Form Guides:**
```
https://www.thegreyhoundrecorder.com.au/form-guides/TRACK_NAME/long-form/RACE_ID/1/
```

## Troubleshooting

### "ERROR - No data scraped"
- Check URL is correct (try it in your browser first)
- Website might be blocking - try again in a few minutes
- Make sure Chrome is installed

### Browser Window Doesn't Close
- This is normal - it closes when scraping completes
- Check the log output in the GUI for progress

### "Module not found" Error
- Run: `pip install -r requirements.txt`

## Tips

✅ **DO:**
- Start with scraping results (historical data)
- Use batch mode for multiple races
- Wait 3-5 seconds between scrapes
- Check the log output to confirm success

❌ **DON'T:**
- Scrape too fast (you'll get blocked)
- Close browser window manually (let it finish)
- Use very old URLs (may have different structure)

## File Locations

- **Database**: `greyhound_racing.db`
- **GUI**: `greyhound_racing_gui.py`
- **Batch Scraper**: `batch_scraper.py`
- **Documentation**: `README.md` and `SCRAPING_GUIDE.md`

## Need More Help?

See full documentation:
- [README.md](README.md) - Complete documentation
- [SCRAPING_GUIDE.md](SCRAPING_GUIDE.md) - Detailed scraping guide
