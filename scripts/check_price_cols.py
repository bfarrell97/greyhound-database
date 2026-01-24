import sqlite3

def check():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(GreyhoundEntries)')
    cols = [info[1] for info in cursor.fetchall()]
    
    price_cols = [c for c in cols if any(x in c.lower() for x in ['price', 'odds', 'fixed', 'open', '$'])]
    print("Potential Price Columns Found:")
    print(price_cols)
    conn.close()

if __name__ == "__main__":
    check()
