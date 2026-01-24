import pandas as pd
import sqlite3
import sys
import os

def check_coverage():
    print("Checking Price Column Coverage (Jan 2025 samples)...")
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Get all columns first
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(GreyhoundEntries)")
    all_cols = [info[1] for info in cursor.fetchall()]
    
    # Identify Price Cols
    price_cols = [c for c in all_cols if 'Price' in c and 'Place' not in c]
    print(f"Price Columns Found: {price_cols}")
    
    # Query sample
    query = f"""
    SELECT {', '.join(price_cols)}
    FROM GreyhoundEntries
    LIMIT 10000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\nNon-Null Counts (First 10,000 rows):")
    for c in price_cols:
        count = df[c].count()
        print(f"  {c:<15}: {count} ({count/100:.1f}%)")

if __name__ == "__main__":
    check_coverage()
