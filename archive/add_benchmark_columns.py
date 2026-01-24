"""Add benchmark comparison columns to GreyhoundEntries and RaceMeetings tables"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("ADDING BENCHMARK COMPARISON COLUMNS")
print("=" * 80)

# Add columns to GreyhoundEntries table
print("\n1. Adding columns to GreyhoundEntries table...")
columns_greyhound = [
    ('FinishTimeBenchmarkLengths', 'REAL'),
    ('SplitBenchmarkLengths', 'REAL')
]

for column_name, column_type in columns_greyhound:
    try:
        cursor.execute(f"ALTER TABLE GreyhoundEntries ADD COLUMN {column_name} {column_type}")
        print(f"   [OK] Added column: {column_name}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print(f"   [SKIP] Column already exists: {column_name}")
        else:
            print(f"   [ERROR] {column_name}: {e}")

# Add columns to RaceMeetings table
print("\n2. Adding columns to RaceMeetings table...")
columns_meeting = [
    ('MeetingAvgBenchmarkLengths', 'REAL'),
    ('MeetingSplitAvgBenchmarkLengths', 'REAL')
]

for column_name, column_type in columns_meeting:
    try:
        cursor.execute(f"ALTER TABLE RaceMeetings ADD COLUMN {column_name} {column_type}")
        print(f"   [OK] Added column: {column_name}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print(f"   [SKIP] Column already exists: {column_name}")
        else:
            print(f"   [ERROR] {column_name}: {e}")

conn.commit()
conn.close()

print("\n" + "=" * 80)
print("DONE! Benchmark comparison columns have been added.")
print("=" * 80)
print("\nColumns added:")
print("  GreyhoundEntries:")
print("    - FinishTimeBenchmarkLengths (positive = faster, negative = slower)")
print("    - SplitBenchmarkLengths (positive = faster, negative = slower)")
print("  RaceMeetings:")
print("    - MeetingAvgBenchmarkLengths (average for all races at meeting)")
print("    - MeetingSplitAvgBenchmarkLengths (average split for all races)")
print("\nNote: 1 length = 0.07 seconds")
