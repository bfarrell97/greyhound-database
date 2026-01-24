# Benchmark Adjustment Columns - Implementation Summary

## Overview
Added 6 new benchmark adjustment columns to the upcoming races form guide, showing how greyhounds and meetings perform relative to track/distance benchmarks.

## What Was Changed

### 1. Database Query Update ([greyhound_database.py](greyhound_database.py))
Updated `get_greyhound_form()` method (lines 436-463) to include 4 new columns:
- `ge.SplitBenchmarkLengths as GFirstSecADJ`
- `rm.MeetingSplitAvgBenchmarkLengths as MFirstSecADJ`
- `ge.FinishTimeBenchmarkLengths as GOTADJ`
- `rm.MeetingAvgBenchmarkLengths as MOTADJ`

### 2. GUI Headers Update ([greyhound_racing_gui.py](greyhound_racing_gui.py))
Updated `self.upcoming_headers` (lines 200-205) to include 6 new columns:
```python
'First Sec', 'G First Sec ADJ', 'M First Sec ADJ', 'G/M First Sec ADJ',
'OT', 'G OT ADJ', 'M OT ADJ', 'G/M OT ADJ'
```

### 3. Display Logic Update ([greyhound_racing_gui.py](greyhound_racing_gui.py))
Added helper functions (lines 832-849):
- `format_benchmark(val)` - Formats benchmark values as "+X.XXL" or "-X.XXL"
- `calc_gm_diff(g_val, m_val)` - Calculates G - M difference

Updated row construction for both first and subsequent rows (lines 851-914) to populate all 6 columns.

## Column Definitions

### First Section (Split Time) Adjustments
1. **G First Sec ADJ**: Greyhound's split time vs benchmark (positive = faster)
2. **M First Sec ADJ**: Meeting's average split vs benchmark (positive = faster)
3. **G/M First Sec ADJ**: G First Sec ADJ - M First Sec ADJ (track condition adjustment)

### Overall Time Adjustments
4. **G OT ADJ**: Greyhound's finish time vs benchmark (positive = faster)
5. **M OT ADJ**: Meeting's average finish time vs benchmark (positive = faster)
6. **G/M OT ADJ**: G OT ADJ - M OT ADJ (track condition adjustment)

## Benchmark Convention
- **1 length = 0.07 seconds**
- **Positive values = FASTER than benchmark** (better performance)
- **Negative values = SLOWER than benchmark** (worse performance)

## Example Interpretation

For a greyhound with these values:
```
G OT ADJ: -7.64L
M OT ADJ: -5.52L
G/M OT ADJ: -2.11L
```

**Interpretation:**
- The greyhound ran **7.64 lengths slower** than the benchmark time
- The meeting overall was running **5.52 lengths slower** (track was slow that day)
- **Adjusted for track conditions**, the greyhound ran **2.11 lengths slower** than the meeting average
- This shows the greyhound underperformed even when accounting for slow track conditions

Another example:
```
G OT ADJ: +0.06L
M OT ADJ: -8.45L
G/M OT ADJ: +8.51L
```

**Interpretation:**
- The greyhound ran **0.06 lengths faster** than benchmark
- The meeting was running **8.45 lengths slower** (very slow track)
- **Adjusted for track conditions**, the greyhound ran **8.51 lengths faster** than the meeting average
- This shows exceptional performance given the slow track conditions

## Why G/M Matters

The G/M adjustment removes the effect of track conditions (wet track, headwind, etc.) to show the greyhound's true performance relative to other runners that day. This is crucial for:

1. **Fair comparison** - A dog running -5L on a slow day might be better than a dog running +2L on a fast day
2. **Track condition assessment** - Identifies when tracks are running unusually fast or slow
3. **Form analysis** - Shows which dogs perform well regardless of conditions

## Files Modified
1. [greyhound_database.py](greyhound_database.py) - Updated form query
2. [greyhound_racing_gui.py](greyhound_racing_gui.py) - Updated headers and display logic

## Testing
Created [test_form_guide_columns.py](test_form_guide_columns.py) to verify:
- All 6 columns are retrieved from database
- Calculations are correct
- Data displays properly in expected format

## Next Steps
The implementation is complete. When you reload the GUI and view an upcoming race:
1. Load a race card using the Upcoming Races tab
2. The form guide will now show all 6 benchmark adjustment columns
3. Values are formatted as "+X.XXL" or "-X.XXL" for easy reading
4. Empty cells appear where benchmark data is not available
