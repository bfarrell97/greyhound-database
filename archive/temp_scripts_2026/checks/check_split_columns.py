import sqlite3
import pandas as pd

def check_splits():
    conn = sqlite3.connect('greyhound_racing.db')
    print("Checking Split vs TopazSplit1 consistency...")
    
    # Check if they are generally same
    query = """
    SELECT 
        TopazSplit1, Split, COUNT(*) as cnt
    FROM GreyhoundEntries 
    WHERE TopazSplit1 IS NOT NULL 
    GROUP BY TopazSplit1, Split 
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    
    # Check for mismatches
    mismatch_query = """
    SELECT COUNT(*) as Mismatches
    FROM GreyhoundEntries 
    WHERE TopazSplit1 IS NOT NULL 
    AND (ABS(TopazSplit1 - Split) > 0.01 OR Split IS NULL)
    """
    mismatch = pd.read_sql_query(mismatch_query, conn)
    print("\nMismatches (TopazSplit1 != Split):")
    print(mismatch.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    check_splits()
