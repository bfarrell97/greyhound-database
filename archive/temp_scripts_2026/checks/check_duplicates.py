import sqlite3
import pandas as pd

def check_dupes():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Check explicitly for Holly Rose
    print("Checking 'HOLLY ROSE'...")
    query = "SELECT * FROM Greyhounds WHERE GreyhoundName LIKE '%HOLLY ROSE%'"
    df = pd.read_sql_query(query, conn)
    print(df)
    
    # Check for general duplicates (trim/case)
    print("\nChecking for potential duplicates (Normalized Name)...")
    # This might be slow on large DB, but simplified check:
    # We fetch all dogs, normalize names in python, look for collisions
    all_dogs = pd.read_sql_query("SELECT GreyhoundID, GreyhoundName FROM Greyhounds", conn)
    
    norm_map = {}
    dupes = []
    
    for _, row in all_dogs.iterrows():
        name = row['GreyhoundName']
        # Normalize: Trim, Upper, Remove double spaces
        norm = " ".join(name.upper().split())
        
        if norm in norm_map:
            dupes.append((norm, norm_map[norm], row['GreyhoundID'], row['GreyhoundName']))
        else:
            norm_map[norm] = row['GreyhoundID']
            
    print(f"\nFound {len(dupes)} potential duplicates.")
    if dupes:
        print("First 10 duplicates:")
        for d in dupes[:10]:
            print(f"Norm: {d[0]} | ID_A: {d[1]} | ID_B: {d[2]} ({d[3]})")
            
    conn.close()

if __name__ == "__main__":
    check_dupes()
