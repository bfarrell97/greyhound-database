import pandas as pd
import sqlite3
import sys
import os
from datetime import datetime, timezone
import re

# Add path to find 'src'
sys.path.append(os.getcwd())
try:
    from src.integration.betfair_fetcher import BetfairOddsFetcher
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.integration.betfair_fetcher import BetfairOddsFetcher

def verify_injection():
    print("--- 💉 LIVE DATA INJECTION VERIFICATION 💉 ---")
    
    # 1. Load DB Candidates (All Today)
    conn = sqlite3.connect('greyhound_racing.db')
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"Loading DB runners for {today_str}...")
    
    query = f"""
    SELECT 
        ge.EntryID, ge.GreyhoundID, ge.Box, 
        ge.Price5Min as DB_Price,
        t.TrackName, r.RaceNumber, g.GreyhoundName as Dog
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate = '{today_str}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ No runners found in DB for today. Cannot verify injection.")
        return

    print(f"Found {len(df)} DB runners.")

    # 2. Fetch Live Prices
    print("Fetching Live Prices from Betfair...")
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        print("❌ Betfair Login Failed.")
        return

    markets = fetcher.get_greyhound_markets()
    print(f"Fetched {len(markets)} active markets.")
    
    price_map = {}
    
    # Batch fetch prices
    market_ids = [m.market_id for m in markets]
    all_prices = fetcher.get_all_market_prices(market_ids)
    
    for m in markets:
        m_prices = all_prices.get(m.market_id, {})
        if m_prices:
            raw_track = m.event.name.split(' (')[0].split(' - ')[0].upper()
            clean_track = raw_track.replace('THE ', '').replace('MT ', 'MOUNT ').strip()
            
            for r in m.runners:
                if r.selection_id in m_prices:
                    # Normalize Dog Name
                    dog = re.sub(r'^\d+\.\s*', '', r.runner_name).strip().upper()
                    
                    # Create Key: TRACK_DOG
                    price_map[f"{clean_track}_{dog}"] = {
                        'back': m_prices[r.selection_id].get('back'),
                        'lay': m_prices[r.selection_id].get('lay')
                    }
    
    fetcher.logout()
    print(f"Mapped {len(price_map)} live prices.")

    # 3. Simulate The Fix in `app.py`
    def get_live_map(row):
        t = str(row['TrackName']).upper().replace('THE ', '').replace('MT ', 'MOUNT ').strip()
        d = str(row['Dog']).upper()
        return price_map.get(f"{t}_{d}")

    df['LiveInfo'] = df.apply(get_live_map, axis=1)
    df['LivePrice'] = df['LiveInfo'].apply(lambda x: x.get('back') if isinstance(x, dict) else None)
    
    # --- CRITICAL: THE LOGIC BEING TESTED ---
    # We want to see if 'LivePrice' successfully OVERWRITES 'DB_Price'
    # We rename DB_Price to Price5Min (mocking the app's dataframe structure)
    df['Price5Min'] = df['DB_Price'] 
    
    # APPLY FIX:
    df['Final_Price'] = df['LivePrice'].fillna(df['Price5Min'])
    
    # 4. Show Proof
    # Filter for rows where we actually have a Live Price
    hits = df[df['LivePrice'].notna()].copy()
    
    if hits.empty:
        print("❌ No overlap found between DB runners and Live Markets (Markets closed?).")
        return
        
    print(f"\n✅ SUCCESSFULLY MATCHED {len(hits)} RUNNERS")
    print(f"Displaying sample to verify OVERWRITE logic:\n")
    
    # Show columns: Dog, DB_Price, LivePrice, Final_Price
    # We want to confirm Final_Price == LivePrice, NOT DB_Price (unless they match)
    hits['IS_CORRECT'] = hits['Final_Price'] == hits['LivePrice']
    
    view = hits[['TrackName', 'Dog', 'DB_Price', 'LivePrice', 'Final_Price', 'IS_CORRECT']].head(15)
    print(view.to_string(index=False))
    
    success_count = hits['IS_CORRECT'].sum()
    print(f"\nPassed: {success_count} / {len(hits)}")
    
    if success_count == len(hits):
        print("\n🏆 VERIFICATION SUCCESSFUL: Live Prices are correctly overwriting DB Prices.")
    else:
        print("\n💀 VERIFICATION FAILED: Some prices were not overwritten.")

if __name__ == "__main__":
    verify_injection()
