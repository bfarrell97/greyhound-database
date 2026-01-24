import sqlite3
conn = sqlite3.connect('greyhound_racing.db')

# Get all tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

print("All tables and columns:\n")
for t in tables:
    table_name = t[0]
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    col_names = [c[1] for c in cols]
    
    # Check for GM or ADJ in column names
    gm_cols = [c for c in col_names if 'GM' in c.upper() or 'ADJ' in c.upper()]
    
    if gm_cols:
        print(f"*** {table_name}: {gm_cols}")
    else:
        print(f"{table_name}: {col_names}")

conn.close()
