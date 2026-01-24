"""
Greyhound Benchmark Comparison Module
Calculates and compares greyhound performance against track/distance benchmarks
Based on Hong Kong Racing benchmark system
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
import statistics


class GreyhoundBenchmarkComparison:
    """Benchmark comparison handler for greyhound racing"""

    def __init__(self, db_path='greyhound_racing.db'):
        self.db_path = db_path
        self.conn = None

    def get_connection(self):
        """Get or create database connection"""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def calculate_benchmark(self, track_name, distance, grade=None, sample_size=1000):
        """
        Calculate benchmark times for a specific track/distance combination

        Args:
            track_name: Name of the track
            distance: Race distance in meters
            grade: Optional grade filter (e.g., 'A1', 'XM5')
            sample_size: Number of recent races to include in benchmark

        Returns:
            Dictionary with benchmark statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Build query based on whether grade is specified
        if grade:
            query = """
                SELECT
                    ge.FinishTime,
                    ge.Split,
                    r.Distance,
                    r.Grade,
                    t.TrackName
                FROM GreyhoundEntries ge
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                WHERE t.TrackName = ?
                AND r.Distance = ?
                AND r.Grade = ?
                AND ge.FinishTime IS NOT NULL
                AND ge.Position = 1
                ORDER BY rm.MeetingDate DESC
                LIMIT ?
            """
            cursor.execute(query, (track_name, distance, grade, sample_size))
        else:
            query = """
                SELECT
                    ge.FinishTime,
                    ge.Split,
                    r.Distance,
                    r.Grade,
                    t.TrackName
                FROM GreyhoundEntries ge
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                WHERE t.TrackName = ?
                AND r.Distance = ?
                AND ge.FinishTime IS NOT NULL
                AND ge.Position = 1
                ORDER BY rm.MeetingDate DESC
                LIMIT ?
            """
            cursor.execute(query, (track_name, distance, sample_size))

        results = cursor.fetchall()

        if not results:
            return None

        # Calculate statistics
        finish_times = [r['FinishTime'] for r in results if r['FinishTime']]
        splits = [r['Split'] for r in results if r['Split']]

        benchmark = {
            'track_name': track_name,
            'distance': distance,
            'grade': grade,
            'avg_time': statistics.mean(finish_times) if finish_times else None,
            'median_time': statistics.median(finish_times) if finish_times else None,
            'fastest_time': min(finish_times) if finish_times else None,
            'slowest_time': max(finish_times) if finish_times else None,
            'std_dev': statistics.stdev(finish_times) if len(finish_times) > 1 else 0,
            'sample_size': len(finish_times),
            'avg_split': statistics.mean(splits) if splits else None,
            'median_split': statistics.median(splits) if splits else None,
            'fastest_split': min(splits) if splits else None,
            'split_sample_size': len(splits) if splits else 0,
        }

        return benchmark

    def calculate_all_benchmarks(self, sample_size=1000, min_races=5):
        """
        Calculate benchmarks for all track/distance combinations in database
        Only creates benchmarks where there are enough races

        Args:
            sample_size: Number of races to use for each benchmark
            min_races: Minimum number of races required to create a benchmark

        Returns:
            Number of benchmarks created/updated
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get all unique track/distance combinations
        cursor.execute("""
            SELECT DISTINCT
                t.TrackName,
                r.Distance,
                COUNT(*) as race_count
            FROM Races r
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            GROUP BY t.TrackName, r.Distance
            HAVING COUNT(*) >= ?
            ORDER BY t.TrackName, r.Distance
        """, (min_races,))

        combinations = cursor.fetchall()
        benchmarks_created = 0

        for combo in combinations:
            track_name = combo['TrackName']
            distance = combo['Distance']

            # Calculate benchmark for this combination (all grades combined)
            benchmark = self.calculate_benchmark(track_name, distance, None, sample_size)

            if benchmark:
                # Save to database
                self.save_benchmark(benchmark)
                benchmarks_created += 1

                split_info = ""
                if benchmark['avg_split']:
                    split_info = f", AvgSplit: {benchmark['avg_split']:.2f}s (n={benchmark['split_sample_size']})"

                print(f"Created benchmark: {track_name} {distance}m - "
                      f"Avg: {benchmark['avg_time']:.2f}s, "
                      f"Samples: {benchmark['sample_size']}"
                      f"{split_info}")

        conn.commit()
        return benchmarks_created

    def save_benchmark(self, benchmark):
        """Save benchmark to database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # First, try to find existing benchmark
        cursor.execute("""
            SELECT BenchmarkID FROM Benchmarks
            WHERE TrackName = ? AND Distance = ? AND Grade IS ?
        """, (
            benchmark['track_name'],
            benchmark['distance'],
            benchmark['grade']
        ))

        existing = cursor.fetchone()

        if existing:
            # Update existing benchmark
            cursor.execute("""
                UPDATE Benchmarks
                SET AvgTime = ?, MedianTime = ?, FastestTime = ?,
                    SlowestTime = ?, StdDev = ?, SampleSize = ?,
                    AvgSplit = ?, MedianSplit = ?, FastestSplit = ?, SplitSampleSize = ?,
                    DateCreated = CURRENT_TIMESTAMP
                WHERE BenchmarkID = ?
            """, (
                benchmark['avg_time'],
                benchmark['median_time'],
                benchmark['fastest_time'],
                benchmark['slowest_time'],
                benchmark['std_dev'],
                benchmark['sample_size'],
                benchmark.get('avg_split'),
                benchmark.get('median_split'),
                benchmark.get('fastest_split'),
                benchmark.get('split_sample_size'),
                existing[0]
            ))
        else:
            # Insert new benchmark
            cursor.execute("""
                INSERT INTO Benchmarks
                (TrackName, Distance, Grade, AvgTime, MedianTime, FastestTime,
                 SlowestTime, StdDev, SampleSize, AvgSplit, MedianSplit, FastestSplit, SplitSampleSize)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark['track_name'],
                benchmark['distance'],
                benchmark['grade'],
                benchmark['avg_time'],
                benchmark['median_time'],
                benchmark['fastest_time'],
                benchmark['slowest_time'],
                benchmark['std_dev'],
                benchmark['sample_size'],
                benchmark.get('avg_split'),
                benchmark.get('median_split'),
                benchmark.get('fastest_split'),
                benchmark.get('split_sample_size')
            ))

        conn.commit()

    def get_benchmark(self, track_name, distance, grade=None):
        """
        Get benchmark for specific track/distance

        Returns:
            Benchmark dictionary or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if grade:
            cursor.execute("""
                SELECT * FROM Benchmarks
                WHERE TrackName = ? AND Distance = ? AND Grade = ?
            """, (track_name, distance, grade))
        else:
            cursor.execute("""
                SELECT * FROM Benchmarks
                WHERE TrackName = ? AND Distance = ? AND (Grade IS NULL OR Grade = '')
            """, (track_name, distance))

        result = cursor.fetchone()

        if result:
            return dict(result)
        return None

    def calculate_time_adjustment_in_lengths(self, actual_time, benchmark_time, distance):
        """
        Calculate how many lengths faster/slower than benchmark

        For greyhounds, 1 length = approximately 0.08 seconds (adjustable)
        Positive value = slower than benchmark
        Negative value = faster than benchmark

        Args:
            actual_time: The greyhound's actual time
            benchmark_time: The benchmark time
            distance: Race distance (may affect length calculation)

        Returns:
            Lengths above/below benchmark
        """
        if not actual_time or not benchmark_time:
            return None

        time_diff = actual_time - benchmark_time

        # Convert time difference to lengths
        # Standard: 1 length = 0.08 seconds (this can be calibrated)
        seconds_per_length = 0.08
        lengths = time_diff / seconds_per_length

        return round(lengths, 2)

    def compare_greyhound_to_benchmark(self, greyhound_name, track_name, distance):
        """
        Compare a greyhound's performance to track/distance benchmark

        Returns:
            Dictionary with comparison statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get benchmark
        benchmark = self.get_benchmark(track_name, distance)
        if not benchmark:
            return {
                'error': f'No benchmark found for {track_name} {distance}m'
            }

        # Get greyhound's races at this track/distance
        cursor.execute("""
            SELECT
                ge.FinishTime,
                ge.Split,
                ge.Position,
                ge.Margin,
                r.Grade,
                rm.MeetingDate
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE g.GreyhoundName = ?
            AND t.TrackName = ?
            AND r.Distance = ?
            AND ge.FinishTime IS NOT NULL
            ORDER BY rm.MeetingDate DESC
        """, (greyhound_name, track_name, distance))

        races = cursor.fetchall()

        if not races:
            return {
                'error': f'{greyhound_name} has no races at {track_name} {distance}m'
            }

        # Calculate comparisons
        time_diffs = []
        split_diffs = []

        for race in races:
            if race['FinishTime']:
                diff = self.calculate_time_adjustment_in_lengths(
                    race['FinishTime'],
                    benchmark['AvgTime'],
                    distance
                )
                if diff is not None:
                    time_diffs.append(diff)

            if race['Split'] and benchmark.get('AvgSplit'):
                split_diff = self.calculate_time_adjustment_in_lengths(
                    race['Split'],
                    benchmark['AvgSplit'],
                    distance
                )
                if split_diff is not None:
                    split_diffs.append(split_diff)

        comparison = {
            'greyhound_name': greyhound_name,
            'track_name': track_name,
            'distance': distance,
            'races_at_track_distance': len(races),
            'benchmark_avg_time': benchmark['AvgTime'],
            'benchmark_avg_split': benchmark.get('AvgSplit'),
            'avg_variance_lengths': statistics.mean(time_diffs) if time_diffs else None,
            'best_variance_lengths': min(time_diffs) if time_diffs else None,
            'consistency': statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0,
            'avg_split_variance_lengths': statistics.mean(split_diffs) if split_diffs else None,
        }

        return comparison

    def get_greyhound_track_record(self, greyhound_name, track_name):
        """
        Get greyhound's record at a specific track (all distances)

        Returns:
            Dictionary with wins-runs-places stats
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as starts,
                SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN ge.Position = 2 THEN 1 ELSE 0 END) as seconds,
                SUM(CASE WHEN ge.Position = 3 THEN 1 ELSE 0 END) as thirds,
                SUM(CASE WHEN ge.Position <= 3 THEN 1 ELSE 0 END) as places
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE g.GreyhoundName = ?
            AND t.TrackName = ?
        """, (greyhound_name, track_name))

        result = cursor.fetchone()

        if result:
            return {
                'starts': result['starts'],
                'wins': result['wins'],
                'seconds': result['seconds'],
                'thirds': result['thirds'],
                'places': result['places'],
                'win_pct': (result['wins'] / result['starts'] * 100) if result['starts'] > 0 else 0,
                'place_pct': (result['places'] / result['starts'] * 100) if result['starts'] > 0 else 0
            }

        return None

    def get_greyhound_track_distance_record(self, greyhound_name, track_name, distance):
        """
        Get greyhound's record at a specific track and distance

        Returns:
            Dictionary with wins-runs-places stats
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as starts,
                SUM(CASE WHEN ge.Position = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN ge.Position = 2 THEN 1 ELSE 0 END) as seconds,
                SUM(CASE WHEN ge.Position = 3 THEN 1 ELSE 0 END) as thirds,
                SUM(CASE WHEN ge.Position <= 3 THEN 1 ELSE 0 END) as places
            FROM GreyhoundEntries ge
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            WHERE g.GreyhoundName = ?
            AND t.TrackName = ?
            AND r.Distance = ?
        """, (greyhound_name, track_name, distance))

        result = cursor.fetchone()

        if result:
            return {
                'starts': result['starts'],
                'wins': result['wins'],
                'seconds': result['seconds'],
                'thirds': result['thirds'],
                'places': result['places'],
                'win_pct': (result['wins'] / result['starts'] * 100) if result['starts'] > 0 else 0,
                'place_pct': (result['places'] / result['starts'] * 100) if result['starts'] > 0 else 0
            }

        return None

    def get_all_benchmarks(self):
        """Get all benchmarks from database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM Benchmarks
            ORDER BY TrackName, Distance
        """)

        return cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


if __name__ == "__main__":
    # Example usage
    comparison = GreyhoundBenchmarkComparison()

    try:
        # Calculate all benchmarks
        print("Calculating benchmarks for all track/distance combinations...")
        count = comparison.calculate_all_benchmarks(sample_size=1000, min_races=5)
        print(f"\nCreated {count} benchmarks")

        # Show all benchmarks
        print("\n" + "=" * 80)
        print("All Benchmarks:")
        print("=" * 80)
        benchmarks = comparison.get_all_benchmarks()

        for b in benchmarks:
            split_info = ""
            if b['AvgSplit']:
                split_info = f", AvgSplit: {b['AvgSplit']:5.2f}s (n={b['SplitSampleSize'] or 0:2})"

            print(f"{b['TrackName']:20} {b['Distance']:4}m - "
                  f"Avg: {b['AvgTime']:6.2f}s, "
                  f"Fastest: {b['FastestTime']:6.2f}s, "
                  f"Samples: {b['SampleSize']:3}"
                  f"{split_info}")

        # Now update benchmark comparisons for all entries
        print("\n" + "=" * 80)
        print("Updating benchmark comparisons...")
        print("=" * 80)
        from update_benchmark_comparisons import calculate_benchmark_comparisons
        calculate_benchmark_comparisons()

    finally:
        comparison.close()
