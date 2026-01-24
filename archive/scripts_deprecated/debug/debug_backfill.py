import sqlite3
import re
import pandas as pd

DB_PATH = 'greyhound_racing.db'

def inspect():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("--- LiveBets (Sample) ---")
    cursor.execute("SELECT BetID, MarketName, MeetingDate, RaceTime FROM LiveBets LIMIT 5")
    rows = cursor.fetchall()
    for r in rows:
        print(r)
        
        # Test Regex
        m_name = r[1]
        date_val = r[2]
        match = re.search(r'(.+) R(\d+)', m_name)
        if match:
            track = match.group(1).strip()
            race_num = match.group(2)
            print(f"  -> Parsed: Track='{track}', Race='{race_num}', Date='{date_val}'")
            
            # Check Races
            cursor.execute("SELECT RaceTime, Date FROM Races WHERE Track=? AND RaceNumber=? LIMIT 1", (track, race_num))
            race_res = cursor.fetchall()
            print(f"  -> Lookup in Races (Track+RaceNum): {race_res}")
            
            # Check with Date
            cursor.execute("SELECT RaceTime FROM Races WHERE Track=? AND RaceNumber=? AND Date=?", (track, race_num, date_val))
            full_res = cursor.fetchone()
            print(f"  -> Full Lookup (Track+RaceNum+Date): {full_res}")
        else:
            print("  -> Regex No Match")

    print("\n--- Races (Sample by Track) ---")
    if rows:
        track_sample = rows[0][1].split(' R')[0]
        cursor.execute("SELECT * FROM Races WHERE Track=? LIMIT 3", (track_sample,))
        print(cursor.fetchall())

    conn.close()

if __name__ == "__main__":
    inspect()
