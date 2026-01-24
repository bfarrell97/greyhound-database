"""Add split time columns to existing Benchmarks table"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Adding split time columns to Benchmarks table...")

# Add split time columns if they don't exist
columns_to_add = [
    ('AvgSplit', 'REAL'),
    ('MedianSplit', 'REAL'),
    ('FastestSplit', 'REAL'),
    ('SplitSampleSize', 'INTEGER')
]

for column_name, column_type in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE Benchmarks ADD COLUMN {column_name} {column_type}")
        print(f"[OK] Added column: {column_name}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print(f"[SKIP] Column already exists: {column_name}")
        else:
            print(f"[ERROR] {column_name}: {e}")

conn.commit()
conn.close()

print("\nDone! Split time columns have been added to the Benchmarks table.")
