"""
Calculate and update benchmark comparisons for all greyhound entries and meetings
This script compares each performance against track/distance benchmarks
1 length = 0.07 seconds
Positive values = faster than benchmark (better)
Negative values = slower than benchmark (worse)
"""

import sqlite3
from greyhound_benchmark_comparison import GreyhoundBenchmarkComparison

SECONDS_PER_LENGTH = 0.07

def calculate_benchmark_comparisons():
    """Calculate benchmark comparisons for all entries and meetings"""

    db_path = 'greyhound_racing.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Load all benchmarks into memory to avoid database locking
    cursor.execute("SELECT TrackName, Distance, AvgTime, AvgSplit FROM Benchmarks")
    benchmarks_data = cursor.fetchall()

    # Create a dictionary for fast lookup: (track, distance) -> benchmark
    benchmarks = {}
    for b in benchmarks_data:
        key = (b['TrackName'], b['Distance'])
        benchmarks[key] = {
            'AvgTime': b['AvgTime'],
            'AvgSplit': b['AvgSplit']
        }

    print("=" * 80)
    print("CALCULATING BENCHMARK COMPARISONS")
    print("=" * 80)
    print(f"Note: 1 length = {SECONDS_PER_LENGTH} seconds")
    print("Positive = faster than benchmark, Negative = slower")
    print("")

    # Step 1: Update GreyhoundEntries
    print("Step 1: Calculating benchmark comparisons for all greyhound entries...")

    cursor.execute("""
        SELECT
            ge.EntryID,
            ge.FinishTime,
            ge.Split,
            t.TrackName,
            r.Distance
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE ge.FinishTime IS NOT NULL
    """)

    entries = cursor.fetchall()
    print(f"Found {len(entries)} entries to process")

    updated_finish = 0
    updated_split = 0

    for i, entry in enumerate(entries, 1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(entries)} entries...")

        entry_id = entry['EntryID']
        finish_time = entry['FinishTime']
        split_time = entry['Split']
        track_name = entry['TrackName']
        distance = entry['Distance']

        # Get benchmark for this track/distance from our dictionary
        benchmark_key = (track_name, distance)
        benchmark = benchmarks.get(benchmark_key)

        if not benchmark:
            continue

        # Calculate finish time comparison
        finish_benchmark_lengths = None
        if finish_time and benchmark.get('AvgTime'):
            # Negative time_diff means faster (better)
            time_diff = benchmark['AvgTime'] - finish_time
            finish_benchmark_lengths = time_diff / SECONDS_PER_LENGTH
            updated_finish += 1

        # Calculate split comparison
        split_benchmark_lengths = None
        if split_time and benchmark.get('AvgSplit'):
            time_diff = benchmark['AvgSplit'] - split_time
            split_benchmark_lengths = time_diff / SECONDS_PER_LENGTH
            updated_split += 1

        # Update entry
        cursor.execute("""
            UPDATE GreyhoundEntries
            SET FinishTimeBenchmarkLengths = ?,
                SplitBenchmarkLengths = ?
            WHERE EntryID = ?
        """, (finish_benchmark_lengths, split_benchmark_lengths, entry_id))

    conn.commit()
    print(f"  Updated {updated_finish} finish time comparisons")
    print(f"  Updated {updated_split} split comparisons")

    # Step 2: Update RaceMeetings
    print("\nStep 2: Calculating meeting-level benchmark averages...")

    cursor.execute("SELECT MeetingID FROM RaceMeetings")
    meetings = cursor.fetchall()
    print(f"Found {len(meetings)} meetings to process")

    updated_meetings = 0

    for meeting in meetings:
        meeting_id = meeting['MeetingID']

        # Calculate average benchmark lengths for all entries in this meeting
        cursor.execute("""
            SELECT
                AVG(ge.FinishTimeBenchmarkLengths) as avg_finish_benchmark,
                AVG(ge.SplitBenchmarkLengths) as avg_split_benchmark
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            WHERE r.MeetingID = ?
              AND ge.FinishTimeBenchmarkLengths IS NOT NULL
        """, (meeting_id,))

        result = cursor.fetchone()

        if result:
            avg_finish = result['avg_finish_benchmark']
            avg_split = result['avg_split_benchmark']

            cursor.execute("""
                UPDATE RaceMeetings
                SET MeetingAvgBenchmarkLengths = ?,
                    MeetingSplitAvgBenchmarkLengths = ?
                WHERE MeetingID = ?
            """, (avg_finish, avg_split, meeting_id))

            updated_meetings += 1

    conn.commit()
    print(f"  Updated {updated_meetings} meetings")

    # Step 3: Show summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    cursor.execute("""
        SELECT
            COUNT(*) as total_entries,
            COUNT(FinishTimeBenchmarkLengths) as entries_with_finish_benchmark,
            COUNT(SplitBenchmarkLengths) as entries_with_split_benchmark,
            AVG(FinishTimeBenchmarkLengths) as avg_finish_vs_benchmark,
            AVG(SplitBenchmarkLengths) as avg_split_vs_benchmark
        FROM GreyhoundEntries
    """)

    stats = cursor.fetchone()

    print(f"\nGreyhoundEntries:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Entries with finish benchmark: {stats['entries_with_finish_benchmark']}")
    print(f"  Entries with split benchmark: {stats['entries_with_split_benchmark']}")
    if stats['avg_finish_vs_benchmark']:
        print(f"  Average finish vs benchmark: {stats['avg_finish_vs_benchmark']:+.2f} lengths")
    if stats['avg_split_vs_benchmark']:
        print(f"  Average split vs benchmark: {stats['avg_split_vs_benchmark']:+.2f} lengths")

    cursor.execute("""
        SELECT
            COUNT(*) as total_meetings,
            COUNT(MeetingAvgBenchmarkLengths) as meetings_with_benchmark,
            AVG(MeetingAvgBenchmarkLengths) as avg_meeting_benchmark
        FROM RaceMeetings
    """)

    meeting_stats = cursor.fetchone()

    print(f"\nRaceMeetings:")
    print(f"  Total meetings: {meeting_stats['total_meetings']}")
    print(f"  Meetings with benchmark: {meeting_stats['meetings_with_benchmark']}")
    if meeting_stats['avg_meeting_benchmark']:
        print(f"  Average meeting benchmark: {meeting_stats['avg_meeting_benchmark']:+.2f} lengths")

    conn.close()

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    calculate_benchmark_comparisons()
