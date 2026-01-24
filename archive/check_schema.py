import sqlite3
conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

print("="*80)
print("DATABASE SCHEMA - Available Data")
print("="*80)

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\n{table_name}:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

# Check if we have opening odds
print("\n" + "="*80)
print("CHECKING FOR ODDS DATA")
print("="*80)

cursor.execute("SELECT * FROM GreyhoundEntries LIMIT 1")
entry_cols = [desc[0] for desc in cursor.description]
print(f"\nGreyhoundEntries columns: {entry_cols}")

if 'OpeningPrice' in entry_cols or 'OpeningOdds' in entry_cols or 'TABOdds' in entry_cols:
    print("✅ OPENING ODDS ARE AVAILABLE!")
else:
    print("❌ No opening odds in current schema")

# Check Race table for additional info
cursor.execute("SELECT * FROM Races LIMIT 1")
race_cols = [desc[0] for desc in cursor.description]
print(f"\nRaces columns: {race_cols}")

conn.close()
