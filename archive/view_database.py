"""
Database Viewer Script
Simple tool to view contents of Greyhound Racing database
"""

import sqlite3
import sys


def view_database(db_path='greyhound_racing.db'):
    """View database contents"""

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print("=" * 80)
        print(f"Database: {db_path}")
        print("=" * 80)

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"\nFound {len(tables)} tables:\n")

        for table in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            print(f"Table: {table} ({count} rows)")
            print("-" * 80)

            if count > 0:
                # Show first 5 rows
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                rows = cursor.fetchall()

                if rows:
                    # Get column names
                    columns = [description[0] for description in cursor.description]

                    # Print column headers
                    header = " | ".join(f"{col[:15]:15}" for col in columns)
                    print(header)
                    print("-" * len(header))

                    # Print rows
                    for row in rows:
                        values = []
                        for val in row:
                            if val is None:
                                values.append("NULL".ljust(15))
                            else:
                                str_val = str(val)[:15]
                                values.append(str_val.ljust(15))
                        print(" | ".join(values))

                    if count > 5:
                        print(f"... and {count - 5} more rows")
            else:
                print("  (empty)")

            print()

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError:
        print(f"Database file not found: {db_path}")
        print("\nMake sure you have run the scraper to create the database first.")


def view_specific_table(db_path, table_name, limit=20):
    """View specific table contents"""

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' not found in database")
            return

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        print("=" * 80)
        print(f"Table: {table_name} ({count} total rows, showing {min(limit, count)})")
        print("=" * 80)

        if count > 0:
            # Get all rows (limited)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = cursor.fetchall()

            if rows:
                # Get column names
                columns = [description[0] for description in cursor.description]

                # Print each row with column names
                for i, row in enumerate(rows, 1):
                    print(f"\nRow {i}:")
                    for col, val in zip(columns, row):
                        print(f"  {col:20}: {val}")
        else:
            print("(empty)")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")


def search_greyhound(db_path, greyhound_name):
    """Search for a specific greyhound"""

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find greyhound
        cursor.execute("SELECT * FROM Greyhounds WHERE GreyhoundName LIKE ?", (f"%{greyhound_name}%",))
        greyhounds = cursor.fetchall()

        if not greyhounds:
            print(f"No greyhounds found matching '{greyhound_name}'")
            conn.close()
            return

        for greyhound in greyhounds:
            print("=" * 80)
            print(f"Greyhound: {greyhound['GreyhoundName']}")
            print("=" * 80)
            print(f"Starts: {greyhound['Starts']}")
            print(f"Wins: {greyhound['Wins']}")
            print(f"Prizemoney: ${greyhound['Prizemoney']:,.2f}")
            print(f"Win %: {greyhound['WinPercentage']:.1f}%")

            # Get race entries
            cursor.execute("""
                SELECT
                    r.RaceNumber,
                    rm.MeetingDate,
                    t.TrackName,
                    r.Distance,
                    r.Grade,
                    ge.Position,
                    ge.FinishTime,
                    tr.TrainerName
                FROM GreyhoundEntries ge
                JOIN Races r ON ge.RaceID = r.RaceID
                JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
                JOIN Tracks t ON rm.TrackID = t.TrackID
                JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
                WHERE ge.GreyhoundID = ?
                ORDER BY rm.MeetingDate DESC
                LIMIT 10
            """, (greyhound['GreyhoundID'],))

            races = cursor.fetchall()

            if races:
                print(f"\nRecent form (last {len(races)} races):")
                print("-" * 80)
                for race in races:
                    print(f"{race['MeetingDate']:10} | {race['TrackName']:15} | "
                          f"{race['Distance']:4}m | {race['Grade'] or 'N/A':8} | "
                          f"Pos {race['Position'] or '-':2} | {race['FinishTime'] or '-':8} | "
                          f"{race['TrainerName']}")

            print()

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")


def interactive_menu():
    """Interactive menu for database viewing"""

    print("\n" + "=" * 80)
    print("Greyhound Racing Database Viewer")
    print("=" * 80)

    # Check for database file
    import os
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]

    if not db_files:
        print("\nNo database files found in current directory.")
        print("Please run the scraper first to create a database.")
        return

    print(f"\nFound database files: {', '.join(db_files)}")

    if len(db_files) == 1:
        db_path = db_files[0]
    else:
        db_path = input(f"\nEnter database filename (default: {db_files[0]}): ").strip()
        if not db_path:
            db_path = db_files[0]

    while True:
        print("\n" + "-" * 80)
        print("Options:")
        print("1. View all tables summary")
        print("2. View specific table")
        print("3. Search for greyhound")
        print("4. Run custom SQL query")
        print("5. Show table structure")
        print("0. Exit")
        print("-" * 80)

        choice = input("\nEnter choice (0-5): ").strip()

        if choice == '0':
            print("Exiting...")
            break

        elif choice == '1':
            view_database(db_path)

        elif choice == '2':
            table_name = input("Enter table name (e.g., Greyhounds, Races, GreyhoundEntries): ").strip()
            limit = input("Number of rows to show (default: 20): ").strip()
            limit = int(limit) if limit else 20
            view_specific_table(db_path, table_name, limit)

        elif choice == '3':
            greyhound_name = input("Enter greyhound name (or partial name): ").strip()
            search_greyhound(db_path, greyhound_name)

        elif choice == '4':
            print("\nEnter SQL query (or 'cancel' to cancel):")
            query = input("SQL> ").strip()

            if query.lower() != 'cancel':
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()

                    cursor.execute(query)

                    if query.strip().upper().startswith('SELECT'):
                        rows = cursor.fetchall()
                        if rows:
                            columns = [description[0] for description in cursor.description]
                            print("\n" + " | ".join(f"{col:15}" for col in columns))
                            print("-" * (len(columns) * 17))
                            for row in rows:
                                print(" | ".join(f"{str(val)[:15]:15}" for val in row))
                            print(f"\n{len(rows)} rows returned")
                        else:
                            print("No rows returned")
                    else:
                        conn.commit()
                        print(f"Query executed. Rows affected: {cursor.rowcount}")

                    conn.close()

                except sqlite3.Error as e:
                    print(f"SQL Error: {e}")

        elif choice == '5':
            table_name = input("Enter table name: ").strip()
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                if columns:
                    print(f"\nStructure of table '{table_name}':")
                    print("-" * 80)
                    print(f"{'Column':<20} {'Type':<15} {'Not Null':<10} {'Default':<15} {'PK':<5}")
                    print("-" * 80)
                    for col in columns:
                        cid, name, type_, notnull, default, pk = col
                        print(f"{name:<20} {type_:<15} {str(bool(notnull)):<10} {str(default):<15} {str(bool(pk)):<5}")
                else:
                    print(f"Table '{table_name}' not found")

                conn.close()

            except sqlite3.Error as e:
                print(f"Error: {e}")

        else:
            print("Invalid choice")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage:")
            print("  python view_database.py                    # Interactive menu")
            print("  python view_database.py <db_file>          # View database summary")
            print("  python view_database.py <db_file> <table>  # View specific table")
        elif len(sys.argv) == 2:
            view_database(sys.argv[1])
        elif len(sys.argv) == 3:
            view_specific_table(sys.argv[1], sys.argv[2])
    else:
        # Interactive mode
        interactive_menu()
