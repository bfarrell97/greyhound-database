"""Check if runLineCode is available for any dogs in recent races"""

from topaz_api import TopazAPI
from config import TOPAZ_API_KEY
from datetime import datetime, timedelta

api = TopazAPI(TOPAZ_API_KEY)

print("=" * 80)
print("CHECKING runLineCode AVAILABILITY ACROSS RECENT RACES")
print("=" * 80)

# Test with recent date
test_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

print(f"\nTesting bulk endpoint for VIC on {test_date}...")
year, month, day = test_date.split('-')

try:
    bulk_runs = api.get_bulk_runs_by_day('VIC', int(year), int(month), int(day))

    print(f"[OK] Got {len(bulk_runs)} runs")

    # Count how many have runLineCode
    runs_with_runlinecode = 0
    runs_without_runlinecode = 0
    sample_with_code = None
    sample_without_code = None

    for run in bulk_runs:
        if run.get('runLineCode'):
            runs_with_runlinecode += 1
            if not sample_with_code:
                sample_with_code = run
        else:
            runs_without_runlinecode += 1
            if not sample_without_code:
                sample_without_code = run

    print(f"\nResults:")
    print(f"  Runs WITH runLineCode: {runs_with_runlinecode} ({runs_with_runlinecode/len(bulk_runs)*100:.1f}%)")
    print(f"  Runs WITHOUT runLineCode: {runs_without_runlinecode} ({runs_without_runlinecode/len(bulk_runs)*100:.1f}%)")

    if sample_with_code:
        print(f"\nSample run WITH runLineCode:")
        print(f"  Dog: {sample_with_code['dogName']}")
        print(f"  Track: {sample_with_code['trackName']}")
        print(f"  Race: {sample_with_code['raceNumber']}")
        print(f"  runLineCode: {sample_with_code['runLineCode']}")
        print(f"  firstSplitPosition: {sample_with_code.get('firstSplitPosition')}")
        print(f"  secondSplitPosition: {sample_with_code.get('secondSplitPosition')}")

    if sample_without_code:
        print(f"\nSample run WITHOUT runLineCode:")
        print(f"  Dog: {sample_without_code['dogName']}")
        print(f"  Track: {sample_without_code['trackName']}")
        print(f"  Race: {sample_without_code['raceNumber']}")
        print(f"  runLineCode: {sample_without_code.get('runLineCode')}")
        print(f"  firstSplitPosition: {sample_without_code.get('firstSplitPosition')}")
        print(f"  secondSplitPosition: {sample_without_code.get('secondSplitPosition')}")

    # Check if pattern is related to track
    print(f"\nBreakdown by track:")
    track_stats = {}
    for run in bulk_runs:
        track = run.get('trackName')
        if track not in track_stats:
            track_stats[track] = {'with': 0, 'without': 0}

        if run.get('runLineCode'):
            track_stats[track]['with'] += 1
        else:
            track_stats[track]['without'] += 1

    for track, stats in sorted(track_stats.items()):
        total = stats['with'] + stats['without']
        with_pct = stats['with'] / total * 100 if total > 0 else 0
        print(f"  {track}: {stats['with']}/{total} ({with_pct:.0f}%) have runLineCode")

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nIf runLineCode is mostly empty, the issue is with the GRV API data itself,")
print("not with the scraper or database import scripts.")
