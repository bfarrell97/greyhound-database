"""
Update greyhound statistics from race entries
Run this after importing data to recalculate wins, starts, etc.
"""

from greyhound_database import GreyhoundDatabase

def main():
    print("=" * 80)
    print("UPDATING GREYHOUND STATISTICS")
    print("=" * 80)

    db = GreyhoundDatabase('greyhound_racing.db')

    print("\nRecalculating stats from race entries...")
    count = db.update_greyhound_stats()

    print(f"\nUpdated {count} greyhounds")

    # Show a sample
    print("\n" + "=" * 80)
    print("SAMPLE OF UPDATED STATS")
    print("=" * 80)

    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            GreyhoundName,
            Starts,
            Wins,
            Seconds,
            Thirds,
            WinPercentage,
            PlacePercentage,
            BestTime
        FROM Greyhounds
        WHERE Starts > 0
        ORDER BY Wins DESC, Starts DESC
        LIMIT 20
    """)

    results = cursor.fetchall()

    print(f"\n{'Greyhound':<30} {'Starts':>6} {'Wins':>5} {'2nds':>5} {'3rds':>5} {'Win%':>6} {'Place%':>7} {'Best':>8}")
    print("-" * 100)

    for row in results:
        name, starts, wins, seconds, thirds, win_pct, place_pct, best_time = row
        best_str = f"{best_time:.2f}" if best_time else "N/A"
        print(f"{name:<30} {starts:>6} {wins:>5} {seconds:>5} {thirds:>5} {win_pct:>6.1f} {place_pct:>7.1f} {best_str:>8}")

    db.close()

    print("\n" + "=" * 80)
    print("STATS UPDATE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
