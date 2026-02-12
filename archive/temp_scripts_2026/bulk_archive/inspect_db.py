import sqlite3

DB_PATH = 'greyhound_racing.db'

def list_tables():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables found:")
        for t in tables:
            print(t[0])
            
            # If table sounds like bets, describe it
            if 'bet' in t[0].lower() or 'order' in t[0].lower():
                print(f"--- Schema for {t[0]} ---")
                cursor.execute(f"PRAGMA table_info({t[0]})")
                cols = cursor.fetchall()
                for c in cols:
                    print(c)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_tables()
