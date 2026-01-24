import sqlite3
import pandas as pd

def check_prizemoney_progression():
    conn = sqlite3.connect('greyhound_racing.db')
    
    # Get a dog with many races
    query = '''
    SELECT GreyhoundID, COUNT(*) as cnt 
    FROM GreyhoundEntries 
    WHERE CareerPrizeMoney > 10000
    GROUP BY GreyhoundID 
    ORDER BY cnt DESC 
    LIMIT 1
    '''
    try:
        row = conn.execute(query).fetchone()
        if not row:
            print("No suitable dogs found.")
            return
            
        dog_id = row[0]
        
        # Get progression
        prog_query = f'''
        SELECT 
            rm.MeetingDate, 
            ge.CareerPrizeMoney, 
            ge.PrizeMoney as RacePrize
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        WHERE ge.GreyhoundID = {dog_id}
        ORDER BY rm.MeetingDate ASC
        '''
        
        df = pd.read_sql_query(prog_query, conn)
        
        print(f"Checking Prize Money Progression for GreyhoundID {dog_id}")
        print("-" * 60)
        print(f"{'Date':<12} {'CareerMoney':<15} {'RacePrize':<10} {'Diff':<10}")
        print("-" * 60)
        
        prev_career = 0
        for i, row in df.iterrows():
            career = row['CareerPrizeMoney']
            prize = row['RacePrize'] if row['RacePrize'] else 0
            diff = career - prev_career
            
            print(f"{row['MeetingDate']:<12} {career:<15,.2f} {prize:<10,.2f} {diff:<10,.2f}")
            prev_career = career
            
        print("-" * 60)
        
        # Check for static value
        unique_values = df['CareerPrizeMoney'].nunique()
        print(f"\nUnique CareerPrizeMoney values: {unique_values} (out of {len(df)} races)")
        
        if unique_values < len(df) * 0.5:
            print("WARNING: CareerPrizeMoney seems static or low variance! Potential LEAK (using final career money?).")
        else:
            print("CareerPrizeMoney changes over time. Likely correct snapshot.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_prizemoney_progression()
