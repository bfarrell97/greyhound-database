"""Test benchmark comparison calculations"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("=" * 100)
print("TESTING BENCHMARK COMPARISONS")
print("=" * 100)

# Get a sample of entries with benchmark comparisons
cursor.execute("""
    SELECT
        g.GreyhoundName,
        t.TrackName,
        r.Distance,
        ge.FinishTime,
        ge.Split,
        ge.FinishTimeBenchmarkLengths,
        ge.SplitBenchmarkLengths,
        b.AvgTime as BenchmarkTime,
        b.AvgSplit as BenchmarkSplit
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Benchmarks b ON t.TrackName = b.TrackName AND r.Distance = b.Distance
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
    LIMIT 10
""")

entries = cursor.fetchall()

print(f"\nSample of 10 entries with benchmark comparisons:")
print("-" * 100)
print(f"{'Dog':<20} {'Track':<15} {'Dist':<5} {'Time':<7} {'Benchmark':<9} {'Diff Lengths':<12}")
print("-" * 100)

for entry in entries:
    dog = entry['GreyhoundName'][:19]
    track = entry['TrackName'][:14]
    dist = f"{entry['Distance']}m"
    time = f"{entry['FinishTime']:.2f}s" if entry['FinishTime'] else "-"
    benchmark = f"{entry['BenchmarkTime']:.2f}s" if entry['BenchmarkTime'] else "-"
    diff_lengths = f"{entry['FinishTimeBenchmarkLengths']:+.2f}L" if entry['FinishTimeBenchmarkLengths'] is not None else "-"

    print(f"{dog:<20} {track:<15} {dist:<5} {time:<7} {benchmark:<9} {diff_lengths:<12}")

# Test meeting-level benchmarks
print("\n" + "=" * 100)
print("MEETING-LEVEL BENCHMARKS")
print("=" * 100)

cursor.execute("""
    SELECT
        t.TrackName,
        rm.MeetingDate,
        rm.MeetingAvgBenchmarkLengths,
        rm.MeetingSplitAvgBenchmarkLengths,
        COUNT(DISTINCT r.RaceID) as num_races
    FROM RaceMeetings rm
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Races r ON rm.MeetingID = r.MeetingID
    WHERE rm.MeetingAvgBenchmarkLengths IS NOT NULL
    ORDER BY rm.MeetingDate DESC
    LIMIT 10
""")

meetings = cursor.fetchall()

print(f"\nSample of 10 recent meetings with benchmark comparisons:")
print("-" * 100)
print(f"{'Track':<20} {'Date':<12} {'Races':<6} {'Avg vs Bench':<15} {'Split vs Bench':<15}")
print("-" * 100)

for meeting in meetings:
    track = meeting['TrackName'][:19]
    date = meeting['MeetingDate']
    num_races = meeting['num_races']
    avg_bench = f"{meeting['MeetingAvgBenchmarkLengths']:+.2f}L" if meeting['MeetingAvgBenchmarkLengths'] is not None else "-"
    split_bench = f"{meeting['MeetingSplitAvgBenchmarkLengths']:+.2f}L" if meeting['MeetingSplitAvgBenchmarkLengths'] is not None else "-"

    print(f"{track:<20} {date:<12} {num_races:<6} {avg_bench:<15} {split_bench:<15}")

# Verify calculation manually for one entry
print("\n" + "=" * 100)
print("MANUAL VERIFICATION OF ONE ENTRY")
print("=" * 100)

cursor.execute("""
    SELECT
        g.GreyhoundName,
        t.TrackName,
        r.Distance,
        ge.FinishTime,
        ge.FinishTimeBenchmarkLengths,
        b.AvgTime as BenchmarkTime
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Benchmarks b ON t.TrackName = b.TrackName AND r.Distance = b.Distance
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
    LIMIT 1
""")

test_entry = cursor.fetchone()

if test_entry:
    dog = test_entry['GreyhoundName']
    track = test_entry['TrackName']
    distance = test_entry['Distance']
    finish_time = test_entry['FinishTime']
    stored_lengths = test_entry['FinishTimeBenchmarkLengths']
    benchmark_time = test_entry['BenchmarkTime']

    # Manual calculation: (benchmark - actual) / 0.07
    time_diff = benchmark_time - finish_time
    calculated_lengths = time_diff / 0.07

    print(f"Dog: {dog}")
    print(f"Track/Distance: {track} {distance}m")
    print(f"Actual time: {finish_time:.2f}s")
    print(f"Benchmark time: {benchmark_time:.2f}s")
    print(f"Time difference: {time_diff:+.4f}s")
    print(f"Calculated lengths: {calculated_lengths:+.2f}L")
    print(f"Stored in database: {stored_lengths:+.2f}L")
    print(f"Match: {'YES' if abs(calculated_lengths - stored_lengths) < 0.01 else 'NO'}")

    if calculated_lengths > 0:
        print(f"\nInterpretation: Dog ran {abs(calculated_lengths):.2f} lengths FASTER than benchmark")
    else:
        print(f"\nInterpretation: Dog ran {abs(calculated_lengths):.2f} lengths SLOWER than benchmark")

conn.close()

print("\n" + "=" * 100)
print("TEST COMPLETE")
print("=" * 100)
