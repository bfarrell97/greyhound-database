"""Test bulk monthly endpoint vs daily to compare speed"""

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
import time

api = TopazAPI(TOPAZ_API_KEY)

print("=" * 80)
print("Comparing bulk endpoints: Daily vs Monthly")
print("=" * 80)

# Test 1: Get a single day
print("\nTest 1: Get VIC data for 2024-12-01 (single day)")
start = time.time()
try:
    runs_day = api.get_bulk_runs_by_day('VIC', 2024, 12, 1)
    elapsed_day = time.time() - start
    print(f"  SUCCESS: {len(runs_day)} runs in {elapsed_day:.2f} seconds")
except Exception as e:
    print(f"  ERROR: {e}")
    runs_day = []
    elapsed_day = 0

# Test 2: Get entire month
print("\nTest 2: Get VIC data for 2024-12 (entire month)")
start = time.time()
try:
    runs_month = api.get_bulk_runs_by_month('VIC', 2024, 12)
    elapsed_month = time.time() - start
    print(f"  SUCCESS: {len(runs_month)} runs in {elapsed_month:.2f} seconds")

    # Calculate runs per day
    days_in_dec = 31
    avg_per_day = len(runs_month) / days_in_dec
    print(f"  Average per day: {avg_per_day:.1f} runs")

except Exception as e:
    print(f"  ERROR: {e}")
    runs_month = []
    elapsed_month = 0

# Test 3: Simulate getting a week of data (7 days) using daily endpoint
print("\nTest 3: Get VIC data for 7 days (2024-12-01 to 2024-12-07) using daily endpoint")
start = time.time()
total_runs_week = 0
try:
    for day in range(1, 8):
        runs = api.get_bulk_runs_by_day('VIC', 2024, 12, day)
        total_runs_week += len(runs)
    elapsed_week = time.time() - start
    print(f"  SUCCESS: {total_runs_week} runs in {elapsed_week:.2f} seconds")
    print(f"  Average per day: {elapsed_week / 7:.2f} seconds")
except Exception as e:
    print(f"  ERROR: {e}")
    elapsed_week = 0

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if runs_month and runs_day:
    print(f"\nMonthly endpoint:")
    print(f"  - Returns entire month in one call")
    print(f"  - {len(runs_month)} runs in {elapsed_month:.2f} seconds")
    print(f"  - Rate: {len(runs_month) / elapsed_month:.0f} runs/second")

    print(f"\nDaily endpoint:")
    print(f"  - Requires 31 calls for a month")
    print(f"  - {len(runs_day)} runs per day in {elapsed_day:.2f} seconds")
    print(f"  - Estimated for full month: {elapsed_day * 31:.2f} seconds")
    print(f"  - Rate: {len(runs_day) / elapsed_day:.0f} runs/second")

    speedup = (elapsed_day * 31) / elapsed_month if elapsed_month > 0 else 0
    print(f"\nMonthly endpoint is approximately {speedup:.1f}x faster for a full month")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if speedup > 3:
        print("\nUse MONTHLY endpoint for bulk imports!")
        print("  - Much faster for historical data")
        print("  - Fewer API calls = less overhead")
        print("\nUse DAILY endpoint for:")
        print("  - Recent data (last few days)")
        print("  - When you only need specific dates")
    else:
        print("\nDaily and monthly endpoints have similar performance")
