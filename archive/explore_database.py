"""
Comprehensive database exploration
Look at all tables, all columns, and sample data
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cursor.fetchall()]

print("="*100)
print("COMPLETE DATABASE STRUCTURE")
print("="*100)

for table in tables:
    print(f"\n{'='*100}")
    print(f"TABLE: {table}")
    print(f"{'='*100}")
    
    # Get column info
    cursor.execute(f'PRAGMA table_info({table})')
    columns = cursor.fetchall()
    
    print(f"\nColumns ({len(columns)}):")
    for col in columns:
        col_id, col_name, col_type, notnull, default, pk = col
        print(f"  {col_name:30} {col_type:15} (PK={pk}, NOT NULL={notnull})")
    
    # Get row count
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    row_count = cursor.fetchone()[0]
    print(f"\nRow count: {row_count:,}")
    
    # Get sample rows
    if row_count > 0:
        print(f"\nFirst 3 rows:")
        cursor.execute(f'SELECT * FROM {table} LIMIT 3')
        rows = cursor.fetchall()
        
        # Get column names for display
        col_names = [c[1] for c in columns]
        
        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")
            for j, val in enumerate(row):
                if val is not None and len(str(val)) > 50:
                    print(f"    {col_names[j]}: {str(val)[:50]}...")
                else:
                    print(f"    {col_names[j]}: {val}")

print("\n" + "="*100)
print("LOOKING FOR EARLY SPEED / SECTIONAL COLUMNS")
print("="*100)

# Search all tables for columns that might contain early speed or sectional data
all_columns = {}
for table in tables:
    cursor.execute(f'PRAGMA table_info({table})')
    cols = cursor.fetchall()
    for col in cols:
        col_name = col[1]
        all_columns[f"{table}.{col_name}"] = col[2]

# Find relevant columns
print("\nColumns that might be early speed / sectional:")
keywords = ['sec', 'speed', 'early', 'first', 'split', 'time', 'adjust', 'adj', 'gm']
for full_col, col_type in sorted(all_columns.items()):
    if any(kw in full_col.lower() for kw in keywords):
        print(f"  {full_col:50} {col_type}")

conn.close()
