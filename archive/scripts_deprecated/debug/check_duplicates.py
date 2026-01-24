import sqlite3
import sys

def check_duplicates():
    try:
        conn = sqlite3.connect('greyhound_racing.db')
        cur = conn.cursor()
        
        print("--- Checking Zipping Lee ---")
        cur.execute("SELECT GreyhoundID, GreyhoundName FROM Greyhounds WHERE GreyhoundName LIKE '%Zipping Lee%'")
        for row in cur.fetchall():
            print(row)
            
        print("\n--- Checking Shima Polly ---")
        cur.execute("SELECT GreyhoundID, GreyhoundName FROM Greyhounds WHERE GreyhoundName LIKE '%Shima Polly%'")
        for row in cur.fetchall():
            print(row)
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_duplicates()
