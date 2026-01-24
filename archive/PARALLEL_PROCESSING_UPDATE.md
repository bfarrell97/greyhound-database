# Parallel Processing Update - Bulk Import Script

## Summary
Added multiprocessing support to [populate_historical_data_bulk.py](populate_historical_data_bulk.py) to dramatically speed up data imports by processing multiple states simultaneously.

## Performance Improvements

### Before (Sequential Processing)
- Processes one state at a time
- Example: 7 states × 3 months = 21 API calls in sequence
- If each API call takes 2 seconds: **42 seconds total**

### After (Parallel Processing)
- Processes all states simultaneously for each month
- Example: 7 states × 3 months = 21 API calls, but 7 at a time
- If each API call takes 2 seconds: **6 seconds total** (7x speedup!)

## Changes Made

### 1. Thread-Safe Statistics Tracking

**Added threading imports** (line 17):
```python
from threading import Lock
```

**Added statistics lock** (line 29):
```python
self.stats_lock = Lock()  # Thread-safe stats updates
```

### 2. Parallel Processing in `populate_date_range()`

**Updated method** (lines 31-104) to use ThreadPoolExecutor:
```python
# Process all states in parallel for this month
with ThreadPoolExecutor(max_workers=len(states)) as executor:
    futures = {}
    for state in states:
        month_count += 1
        future = executor.submit(self.populate_month_parallel, state, year, month, month_count, total_months)
        futures[future] = state

    # Process results as they complete
    for future in as_completed(futures):
        state = futures[future]
        try:
            result = future.result()
            if result:
                runs_imported, races_imported, errors = result
                with self.stats_lock:
                    self.stats['runs_imported'] += runs_imported
                    self.stats['races_imported'] += races_imported
                    self.stats['errors'] += errors
                    self.stats['months_processed'] += 1
```

### 3. New Thread-Safe Import Method

**Added `populate_month_parallel()`** (lines 106-177):
- Creates thread-local database connection for SQLite thread safety
- Fetches and processes data for one state/month combination
- Returns statistics tuple instead of updating shared stats directly
- Properly closes database connection when done

Key differences from original `populate_month()`:
```python
# Create thread-local database connection (SQLite thread safety)
db = GreyhoundDatabase()

# ... process data ...

# Return statistics instead of updating directly
return (runs_processed, races_processed, errors)
```

## How It Works

### 1. Month-by-Month Parallelism
The script processes each month sequentially, but within each month all states are fetched in parallel:

```
Month 1 (September 2025):
  VIC, NSW, QLD, SA, WA, TAS, NZ - all fetched simultaneously

Month 2 (October 2025):
  VIC, NSW, QLD, SA, WA, TAS, NZ - all fetched simultaneously

Month 3 (November 2025):
  VIC, NSW, QLD, SA, WA, TAS, NZ - all fetched simultaneously
```

### 2. Thread Safety
Each thread gets its own:
- Database connection (SQLite requirement)
- API client (already thread-safe)
- Local statistics (merged at the end)

Shared statistics are protected by `Lock`:
```python
with self.stats_lock:
    self.stats['runs_imported'] += runs_imported
```

### 3. Error Handling
- Each thread handles its own errors
- Errors don't stop other threads
- All errors are tracked and reported

## Expected Performance

### API Call Time Reduction
- **7 states**: ~7x faster for API calls
- **3 states**: ~3x faster for API calls
- **2 states**: ~2x faster for API calls

### Real-World Example
Importing 3 months of data for all 7 states:

**Old version (sequential):**
- 21 API calls at 2 seconds each = 42 seconds for API calls
- Plus database processing time
- **Total: ~60-90 seconds**

**New version (parallel):**
- 3 months × (2 seconds per API call) = 6 seconds for API calls
- Plus database processing time (same as before)
- **Total: ~15-25 seconds**

**Speedup: 3-4x faster overall!**

## User Interface Changes

The script now shows parallel processing in the header:
```
Populating data from 2025-09-01 to 2025-11-30
States: VIC, NSW, QLD, SA, WA, TAS, NZ
Total months to process: 3 months × 7 states = 21 API calls
Parallel processing: 7 states at a time
================================================================================
```

## Technical Details

### SQLite Thread Safety
Each thread must have its own database connection:
```python
# WRONG - shared connection
success = self.db.import_results_data(...)

# RIGHT - thread-local connection
db = GreyhoundDatabase()
success = db.import_results_data(...)
db.close()
```

### ThreadPoolExecutor
- Uses Python's `concurrent.futures.ThreadPoolExecutor`
- `max_workers=len(states)` - one thread per state
- Threads are automatically managed and cleaned up

### Statistics Merging
Results from each thread are collected and merged:
```python
result = future.result()
if result:
    runs_imported, races_imported, errors = result
    with self.stats_lock:
        self.stats['runs_imported'] += runs_imported
        self.stats['races_imported'] += races_imported
        self.stats['errors'] += errors
```

## Files Modified

1. [populate_historical_data_bulk.py](populate_historical_data_bulk.py)
   - Added `populate_month_parallel()` method
   - Updated `populate_date_range()` to use ThreadPoolExecutor
   - Added thread-safe statistics tracking with Lock

## Testing

The parallel version has been tested with:
- Multiple states (VIC, NSW, QLD, SA, WA, TAS, NZ)
- Multiple months
- Error handling (API failures, database errors)
- Thread safety (no race conditions in statistics)

## Usage

No changes to user interface! Just run the script as before:

```bash
python populate_historical_data_bulk.py
```

The script will automatically:
1. Ask for start/end dates
2. Ask for states to import
3. Process all states in parallel
4. Show faster completion times

## Benefits

1. **Much Faster**: 3-7x faster for API calls
2. **Same Reliability**: All error handling preserved
3. **Thread-Safe**: No race conditions or data corruption
4. **Easy to Use**: No interface changes needed
5. **Scalable**: More states = more speedup

## Next Steps

The parallel processing is complete and ready to use. When you run the script:
1. You'll see "Parallel processing: X states at a time" in the header
2. All states for each month will be fetched simultaneously
3. Import times will be significantly faster
4. Statistics will be accurately tracked across all threads
