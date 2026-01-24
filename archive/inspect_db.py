import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print('Tables:', tables)

# Check GreyhoundEntries structure
cursor.execute('PRAGMA table_info(GreyhoundEntries)')
cols = [(c[1], c[2]) for c in cursor.fetchall()]
print('\nGreyhoundEntries columns:')
for col, type_ in cols:
    print(f'  {col}: {type_}')

# Check if Rating column has any data
cursor.execute('SELECT COUNT(*), COUNT(Rating), COUNT(DISTINCT Rating) FROM GreyhoundEntries')
result = cursor.fetchone()
print(f'\nRating column stats: Total={result[0]}, Non-null={result[1]}, Distinct={result[2]}')

# Check EarlySpeed
cursor.execute('SELECT COUNT(*), COUNT(EarlySpeed), COUNT(DISTINCT EarlySpeed) FROM GreyhoundEntries')
result = cursor.fetchone()
print(f'EarlySpeed column stats: Total={result[0]}, Non-null={result[1]}, Distinct={result[2]}')

# Sample some rows
cursor.execute('SELECT Box, Weight, Rating, EarlySpeed FROM GreyhoundEntries LIMIT 10')
print('\nSample rows:')
for row in cursor.fetchall():
    print(row)
