"""
Check Database Statistics
Quick script to check database contents and statistics
"""

import sqlite3
import sys


def check_database(db_path='greyhound_racing.db'):
    """Check database statistics"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("=" * 80)
        print("Greyhound Racing Database Statistics")
        print("=" * 80)
        print(f"Database: {db_path}\n")

        # Greyhounds
        cursor.execute("SELECT COUNT(*) FROM Greyhounds")
        greyhound_count = cursor.fetchone()[0]
        print(f"Total Greyhounds:        {greyhound_count:,}")

        # Trainers
        cursor.execute("SELECT COUNT(*) FROM Trainers")
        trainer_count = cursor.fetchone()[0]
        print(f"Total Trainers:          {trainer_count:,}")

        # Owners
        cursor.execute("SELECT COUNT(*) FROM Owners")
        owner_count = cursor.fetchone()[0]
        print(f"Total Owners:            {owner_count:,}")

        # Tracks
        cursor.execute("SELECT COUNT(*) FROM Tracks")
        track_count = cursor.fetchone()[0]
        print(f"Total Tracks:            {track_count:,}")

        # Race Meetings
        cursor.execute("SELECT COUNT(*) FROM RaceMeetings")
        meeting_count = cursor.fetchone()[0]
        print(f"Total Race Meetings:     {meeting_count:,}")

        # Races
        cursor.execute("SELECT COUNT(*) FROM Races")
        race_count = cursor.fetchone()[0]
        print(f"Total Races:             {race_count:,}")

        # Entries
        cursor.execute("SELECT COUNT(*) FROM GreyhoundEntries")
        entry_count = cursor.fetchone()[0]
        print(f"Total Greyhound Entries: {entry_count:,}")

        # Benchmarks
        cursor.execute("SELECT COUNT(*) FROM Benchmarks")
        benchmark_count = cursor.fetchone()[0]
        print(f"Total Benchmarks:        {benchmark_count:,}")

        print("\n" + "=" * 80)
        print("Date Range")
        print("=" * 80)

        cursor.execute("SELECT MIN(MeetingDate), MAX(MeetingDate) FROM RaceMeetings")
        result = cursor.fetchone()
        if result[0]:
            print(f"First Meeting: {result[0]}")
            print(f"Last Meeting:  {result[1]}")

        print("\n" + "=" * 80)
        print("Tracks in Database")
        print("=" * 80)

        cursor.execute("""
            SELECT t.TrackName, t.State, t.Country, COUNT(DISTINCT rm.MeetingID) as meetings
            FROM Tracks t
            LEFT JOIN RaceMeetings rm ON t.TrackID = rm.TrackID
            GROUP BY t.TrackName
            ORDER BY meetings DESC
        """)

        tracks = cursor.fetchall()
        if tracks:
            print(f"{'Track Name':<25} {'State':<10} {'Country':<10} {'Meetings':<10}")
            print("-" * 80)
            for track in tracks:
                print(f"{track[0]:<25} {track[1] or 'N/A':<10} {track[2]:<10} {track[3]:<10}")

        print("\n" + "=" * 80)
        print("Distance Distribution")
        print("=" * 80)

        cursor.execute("""
            SELECT Distance, COUNT(*) as race_count
            FROM Races
            GROUP BY Distance
            ORDER BY Distance
        """)

        distances = cursor.fetchall()
        if distances:
            print(f"{'Distance':<15} {'Races':<10}")
            print("-" * 80)
            for dist in distances:
                print(f"{dist[0]}m{'':<11} {dist[1]:,}")

        print("\n" + "=" * 80)
        print("Top 10 Greyhounds by Starts")
        print("=" * 80)

        cursor.execute("""
            SELECT GreyhoundName, Starts, Wins, WinPercentage
            FROM Greyhounds
            ORDER BY Starts DESC
            LIMIT 10
        """)

        greyhounds = cursor.fetchall()
        if greyhounds:
            print(f"{'Greyhound':<25} {'Starts':<10} {'Wins':<10} {'Win %':<10}")
            print("-" * 80)
            for dog in greyhounds:
                print(f"{dog[0]:<25} {dog[1]:<10} {dog[2]:<10} {dog[3]:.1f}%")

        print("\n" + "=" * 80)
        print("Benchmarks by Track")
        print("=" * 80)

        cursor.execute("""
            SELECT TrackName, COUNT(*) as benchmark_count
            FROM Benchmarks
            GROUP BY TrackName
            ORDER BY benchmark_count DESC
        """)

        bench_tracks = cursor.fetchall()
        if bench_tracks:
            print(f"{'Track Name':<25} {'Benchmarks':<15}")
            print("-" * 80)
            for bt in bench_tracks:
                print(f"{bt[0]:<25} {bt[1]:<15}")
        else:
            print("No benchmarks calculated yet. Run 'Calculate Benchmarks' from GUI or:")
            print("  python greyhound_racing_gui.py")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError:
        print(f"Database file not found: {db_path}")
        print("\nMake sure you have run the scraper to create the database first.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'greyhound_racing.db'
    check_database(db_path)
