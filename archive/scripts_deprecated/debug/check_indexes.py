import sqlite3

try:
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Indexes on GreyhoundEntries:")
    cursor.execute("PRAGMA index_list('GreyhoundEntries')")
    indexes = cursor.fetchall()
    
    for idx in indexes:
        print(f"Index: {idx[1]} (Unique: {idx[2]})")
        cursor.execute(f"PRAGMA index_info('{idx[1]}')")
        cols = cursor.fetchall()
        col_names = [c[2] for c in cols]
        print(f"  Columns: {col_names}")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
