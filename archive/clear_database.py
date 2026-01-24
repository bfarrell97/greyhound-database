"""
Clear Database Script
Clears all records from the greyhound racing database
"""

import sqlite3
import sys


def clear_database(db_path='greyhound_racing.db', confirm=True):
    """Clear all records from database tables"""

    if confirm:
        response = input(f"Are you sure you want to clear all data from {db_path}? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Clearing database tables...")

        # Clear tables in order (respecting foreign key constraints)
        tables = [
            'Adjustments',
            'SectionalTimes',
            'GreyhoundEntries',
            'Races',
            'RaceMeetings',
            'Benchmarks',
            'Greyhounds',
            'Trainers',
            'Owners',
            # Don't clear Tracks as they are predefined
        ]

        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            count = cursor.rowcount
            print(f"  Cleared {count:,} rows from {table}")

        conn.commit()
        print("\nDatabase cleared successfully!")
        print("Note: Tracks table was not cleared (contains predefined tracks)")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except FileNotFoundError:
        print(f"Database file not found: {db_path}")


def clear_specific_table(db_path, table_name, confirm=True):
    """Clear a specific table"""

    if confirm:
        response = input(f"Are you sure you want to clear table '{table_name}'? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"DELETE FROM {table_name}")
        count = cursor.rowcount

        conn.commit()
        print(f"Cleared {count:,} rows from {table_name}")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage:")
            print("  python clear_database.py                 # Clear all tables")
            print("  python clear_database.py <table_name>    # Clear specific table")
            print("  python clear_database.py --force         # Clear without confirmation")
        elif sys.argv[1] == '--force':
            clear_database(confirm=False)
        else:
            table_name = sys.argv[1]
            clear_specific_table('greyhound_racing.db', table_name)
    else:
        clear_database()
