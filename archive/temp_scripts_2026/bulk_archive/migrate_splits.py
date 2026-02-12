import sqlite3

def migrate_splits():
    conn = sqlite3.connect('greyhound_racing.db')
    cursor = conn.cursor()
    
    print("Migrating Historical Data: TopazSplit1 -> Split...")
    
    # 1. Count fixable rows
    query_count = """
    SELECT COUNT(*) FROM GreyhoundEntries 
    WHERE Split IS NULL AND TopazSplit1 IS NOT NULL
    """
    count = cursor.execute(query_count).fetchone()[0]
    print(f"Found {count} rows where Split is missing but TopazSplit1 exists.")
    
    if count > 0:
        # 2. Execute Update
        # Also migrate TopazPIR -> FirstSplitPosition
        query_update = """
        UPDATE GreyhoundEntries
        SET 
            Split = TopazSplit1,
            FirstSplitPosition = TopazPIR
        WHERE Split IS NULL AND TopazSplit1 IS NOT NULL
        """
        cursor.execute(query_update)
        conn.commit()
        print(f"Migrated {cursor.rowcount} rows.")
    else:
        print("No migration needed.")
        
    conn.close()

if __name__ == "__main__":
    migrate_splits()
