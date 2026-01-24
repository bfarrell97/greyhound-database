import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import GreyhoundDatabase

def debug_import():
    db = GreyhoundDatabase()
    
    # Target: Bendigo R1 (ID 323356)
    # Dog: Saunders (ID 286171)
    # We will try to update its price to 5.5
    
    print("Attempting to import/update Saunders in Bendigo R1...")
    
    # Mock Data matching the structure expected by import_form_guide_data
    # It expects: track_name, race_number, entries=[...]
    
    mock_entries = [{
        'greyhound_name': 'Saunders',
        'box': 4, # Guessing
        'trainer': 'Unknown',
        'starting_price': 5.5
    }]
    
    try:
        # We use the method directly
        success = db.import_form_guide_data(
            {'entries': mock_entries, 'race_number': 1, 'track_name': 'Bendigo', 'date': '2025-12-13', 'track_key': 'bendigo'},
            '2025-12-13',
            'Bendigo'
        )
        print(f"Import returned: {success}")
        
    except Exception as e:
        print(f"CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    # Verify
    cursor = db.conn.cursor()
    cursor.execute("SELECT StartingPrice FROM GreyhoundEntries WHERE RaceID=323356 AND GreyhoundID=286171") # Accessing known IDs
    # Wait, need to check if IDs are correct. 
    # Saunders ID might be different if duplications happened?
    # I'll check by name.
    cursor.execute("""
        SELECT ge.StartingPrice, g.GreyhoundName 
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID=g.GreyhoundID
        WHERE ge.RaceID=323356 AND g.GreyhoundName='Saunders' COLLATE NOCASE
    """)
    row = cursor.fetchone()
    print(f"Verification: {row}")

if __name__ == "__main__":
    debug_import()
