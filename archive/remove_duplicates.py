"""
Utility to remove duplicate race entries from the database

Duplicates are defined as:
- Same greyhound (GreyhoundID) in the same race (RaceID)

This script will keep the most recent entry and delete older duplicates.
"""

from greyhound_database import GreyhoundDatabase
import sqlite3


def check_for_duplicates():
    """Check and display duplicate entries"""
    db = GreyhoundDatabase('greyhound_racing.db')
    conn = db.get_connection()
    cursor = conn.cursor()

    # Find duplicates
    cursor.execute("""
        SELECT
            r.RaceID,
            rm.MeetingDate,
            t.TrackName,
            r.RaceNumber,
            g.GreyhoundName,
            ge.GreyhoundID,
            COUNT(*) as entry_count,
            GROUP_CONCAT(ge.EntryID) as entry_ids,
            GROUP_CONCAT(ge.Position) as positions
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        GROUP BY ge.RaceID, ge.GreyhoundID
        HAVING COUNT(*) > 1
        ORDER BY rm.MeetingDate DESC, r.RaceNumber
    """)

    duplicates = cursor.fetchall()

    if not duplicates:
        print("✓ No duplicates found in the database!")
        return False

    print(f"\n⚠ Found {len(duplicates)} duplicate entries:\n")
    print(f"{'Date':<12} {'Track':<15} {'Race':<5} {'Greyhound':<25} {'Count':<6} {'EntryIDs':<15} {'Positions'}")
    print("-" * 110)

    for dup in duplicates:
        race_id, date, track, race_num, dog_name, dog_id, count, entry_ids, positions = dup
        print(f"{date:<12} {track:<15} {race_num:<5} {dog_name:<25} {count:<6} {entry_ids:<15} {positions}")

    print(f"\nTotal duplicate entries to remove: {sum(d[6] - 1 for d in duplicates)}")
    return True


def main():
    print("=" * 110)
    print("GREYHOUND DATABASE - DUPLICATE REMOVAL UTILITY")
    print("=" * 110)

    # First, check for duplicates
    has_duplicates = check_for_duplicates()

    if not has_duplicates:
        return

    # Ask for confirmation
    print("\n" + "=" * 110)
    response = input("\nDo you want to remove these duplicates? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Cancelled. No changes made.")
        return

    # Remove duplicates
    print("\nRemoving duplicates...\n")
    db = GreyhoundDatabase('greyhound_racing.db')
    removed = db.remove_duplicate_entries()

    print(f"\n✓ Successfully removed {removed} duplicate entries!")

    # Update greyhound stats after cleanup
    print("\nUpdating greyhound statistics...")
    updated = db.update_greyhound_stats()
    print(f"✓ Updated stats for {updated} greyhounds")

    # Verify no duplicates remain
    print("\nVerifying cleanup...")
    check_for_duplicates()

    print("\n" + "=" * 110)
    print("CLEANUP COMPLETE")
    print("=" * 110)


if __name__ == '__main__':
    main()
