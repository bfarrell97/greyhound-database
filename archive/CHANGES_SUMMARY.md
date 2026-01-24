# Changes Summary

## Changes Made - 2025-11-30

### 1. Multi-Race Scraping Feature ✓

**Problem**: User had to manually scrape each race individually (12 different URLs for a 12-race meeting)

**Solution**: Created automatic multi-race scraping that clicks through all race buttons

**Files Modified**:
- [greyhound_scraper_v2.py](greyhound_scraper_v2.py:461-681) - Added `scrape_all_meeting_results()` method
- [greyhound_racing_gui.py](greyhound_racing_gui.py:96-103) - Added checkbox and multi-race logic
- [QUICK_START.md](QUICK_START.md:20) - Updated instructions

**How It Works**:
1. Loads results URL (e.g., Race 1)
2. Finds all race selector buttons (1-12)
3. Clicks each button and waits for data to load (1.5s)
4. Scrapes each race automatically
5. Saves all races to database

**Usage**:
```bash
# Via GUI
python greyhound_racing_gui.py
# Check "Scrape entire meeting" checkbox

# Via test script
python test_multi_race_scraper.py
```

---

### 2. Greyhound Statistics Fix ✓

**Problem**: All greyhounds showed 0 starts and 0 wins, even though race data was saved

**Root Cause**: `add_or_get_greyhound()` uses `INSERT OR IGNORE`, which doesn't update existing greyhound stats

**Solution**: Created `update_greyhound_stats()` method that calculates stats from race entries

**Files Modified**:
- [greyhound_database.py](greyhound_database.py:621-676) - Added `update_greyhound_stats()` method
- [greyhound_database.py](greyhound_database.py:543-544) - Call stats update after importing results

**Created Files**:
- [update_stats.py](update_stats.py) - Utility to recalculate all greyhound stats

**Stats Calculated**:
- Starts (count of all race entries)
- Wins (count of Position = 1)
- Seconds (count of Position = 2)
- Thirds (count of Position = 3)
- Win Percentage
- Place Percentage
- Best Time (minimum finish time)

**Usage**:
```bash
# Manually update all stats
python update_stats.py
```

**Results**:
```
Greyhound                      Starts  Wins  2nds  3rds   Win%  Place%     Best
----------------------------------------------------------------------------------------------------
Slick Nitro                         2     2     0     0  100.0   100.0    25.47
Cornhill Boomer                     1     1     0     0  100.0   100.0    22.47
Time Perpetuated                    2     0     2     0    0.0   100.0    25.68
```

---

### 3. TkSheet Integration ✓

**Problem**: GUI was using ScrolledText widgets for data display (not user-friendly for tabular data)

**Solution**: Integrated tksheet for professional spreadsheet-like data display

**Files Modified**:
- [greyhound_racing_gui.py](greyhound_racing_gui.py:262-270) - Database Viewer now uses tksheet
- [greyhound_racing_gui.py](greyhound_racing_gui.py:206-214) - Greyhound Analysis now uses tksheet
- [greyhound_racing_gui.py](greyhound_racing_gui.py:555-597) - Updated `view_table()` method
- [greyhound_racing_gui.py](greyhound_racing_gui.py:463-513) - Updated `load_greyhound_form()` method

**Features**:
- Sortable columns
- Resizable columns
- Auto-sizing to fit content
- Scroll horizontally and vertically
- Professional table appearance
- Copy/paste support
- Much better for viewing race data

**Tabs Using TkSheet**:
1. ✓ **Database Viewer** - View any database table
2. ✓ **Greyhound Analysis** - View greyhound form/history
3. ✓ **Upcoming Races** - View upcoming race cards (already had tksheet)

---

## Summary of All Changes

### New Features
1. **Multi-race scraping** - Scrape entire meetings with one click
2. **Auto-updating stats** - Greyhound wins/starts update automatically
3. **TkSheet display** - Professional spreadsheet-like data viewing

### Bug Fixes
1. **Fixed SQLite threading error** - Each thread now creates its own DB connection
2. **Fixed greyhound stats** - Stats now calculated from actual race entries
3. **Fixed race selector detection** - Updated CSS selectors for website structure

### Files Created
- [test_multi_race_scraper.py](test_multi_race_scraper.py) - Test multi-race scraping
- [update_stats.py](update_stats.py) - Recalculate greyhound statistics
- [debug_wins.py](debug_wins.py) - Debug script for checking stats
- [MULTI_RACE_SCRAPING.md](MULTI_RACE_SCRAPING.md) - Multi-race feature documentation
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - This file

### Files Modified
- [greyhound_scraper_v2.py](greyhound_scraper_v2.py) - Added multi-race scraping
- [greyhound_database.py](greyhound_database.py) - Added stats update method
- [greyhound_racing_gui.py](greyhound_racing_gui.py) - Added checkbox, tksheet integration
- [QUICK_START.md](QUICK_START.md) - Updated instructions

---

## Testing Performed

### Multi-Race Scraping
✓ Successfully scraped all 12 races from Ballarat meeting (Nov 29, 2025)
✓ All races saved to database correctly
✓ Race selector buttons detected using `.meeting-events-nav__item`
✓ Wait time optimized to 1.5 seconds per race

### Greyhound Stats
✓ Updated 122 greyhounds from race entries
✓ Slick Nitro correctly shows 2 starts, 2 wins
✓ Stats match actual race results in database
✓ Best times calculated correctly

### TkSheet Display
✓ Database Viewer displays tables correctly
✓ Greyhound Analysis shows form data in grid
✓ Columns auto-size to content
✓ All data displays properly

---

## Known Limitations

1. **Multi-race scraping** only works for results URLs (not form guides)
2. **Form guides** must still be scraped individually (different race IDs)
3. **CloudFlare protection** may occasionally block scraping (retry if needed)
4. **Stats update** runs after every results import (slight performance impact)

---

## Next Steps / Future Enhancements

1. Add benchmark adjustments to Greyhound Analysis display
2. Add search functionality for greyhounds
3. Implement upcoming races form display with historical data
4. Add export to CSV/Excel functionality
5. Add filtering/sorting in tksheet displays
6. Add track condition analysis
7. Add box bias analysis

---

## For Users

### Quick Reference

**Multi-Race Scraping**:
1. Open GUI: `python greyhound_racing_gui.py`
2. Paste results URL
3. Check "Scrape entire meeting"
4. Click "Scrape and Save to Database"

**View Greyhound Form**:
1. Go to "Greyhound Analysis" tab
2. Enter greyhound name (e.g., "Slick Nitro")
3. Click "Load Form"
4. View results in spreadsheet

**View Database**:
1. Go to "Database Viewer" tab
2. Select table from dropdown
3. Click "View Table"
4. Data displays in spreadsheet

**Update Stats Manually**:
```bash
python update_stats.py
```
