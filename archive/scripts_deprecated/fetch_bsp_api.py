"""
Fetch BSP via Betfair API - for recent/today's races
The Betfair API keeps market data for a few days after settlement
"""
import requests
import sqlite3
import json
from datetime import datetime, timedelta
import sys
sys.path.append('src')
from core.config import BETFAIR_APP_KEY

DB_PATH = 'greyhound_racing.db'
SSOID = "z0tnRErkLeSV9BlkatWWYx+/4zhC7dd33p4NBNTx7Pc="
BETTING_API = "https://api.betfair.com/exchange/betting/rest/v1.0/"

def get_headers():
    return {
        "X-Application": BETFAIR_APP_KEY,
        "X-Authentication": SSOID,
        "Content-Type": "application/json"
    }

def list_greyhound_markets(from_date, to_date):
    """Get greyhound racing markets for date range"""
    url = BETTING_API + "listMarketCatalogue/"
    
    params = {
        "filter": {
            "eventTypeIds": ["4339"],  # Greyhound Racing
            "marketCountries": ["AU"],
            "marketTypeCodes": ["WIN"],
            "marketStartTime": {
                "from": from_date.strftime("%Y-%m-%dT00:00:00Z"),
                "to": to_date.strftime("%Y-%m-%dT23:59:59Z")
            }
        },
        "maxResults": "1000",
        "marketProjection": ["RUNNER_DESCRIPTION", "MARKET_START_TIME", "EVENT"]
    }
    
    try:
        resp = requests.post(url, headers=get_headers(), json=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Error {resp.status_code}: {resp.text[:300]}", flush=True)
            return []
    except Exception as e:
        print(f"Request error: {e}", flush=True)
        return []

def get_market_book_with_sp(market_ids):
    """Get market book with SP data"""
    url = BETTING_API + "listMarketBook/"
    
    params = {
        "marketIds": market_ids,
        "priceProjection": {
            "priceData": ["SP_AVAILABLE", "SP_TRADED"]
        }
    }
    
    try:
        resp = requests.post(url, headers=get_headers(), json=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"MarketBook Error {resp.status_code}: {resp.text[:300]}", flush=True)
            return []
    except Exception as e:
        print(f"MarketBook error: {e}", flush=True)
        return []

def main():
    print("="*60, flush=True)
    print("BETFAIR API BSP FETCHER", flush=True)
    print("="*60, flush=True)
    
    # Check for recent settled markets (last 3 days)
    today = datetime.now()
    from_date = today - timedelta(days=3)
    
    print(f"\nFetching AU greyhound markets from {from_date.date()} to {today.date()}...", flush=True)
    
    markets = list_greyhound_markets(from_date, today)
    print(f"Found {len(markets)} markets", flush=True)
    
    if not markets:
        print("No markets found. API may not have historical data.", flush=True)
        return
    
    # Show sample markets
    print("\nSample markets:", flush=True)
    for m in markets[:5]:
        event = m.get('event', {})
        venue = event.get('venue', 'Unknown')
        name = m.get('marketName', 'Unknown')
        start = m.get('marketStartTime', '')
        print(f"  {venue} - {name} ({start[:16]})", flush=True)
    
    # Build runner name to market mapping
    print("\nBuilding runner lookup...", flush=True)
    runner_lookup = {}  # (venue, date, runner_name) -> (market_id, selection_id)
    
    for m in markets:
        market_id = m['marketId']
        event = m.get('event', {})
        venue = event.get('venue', '').upper()
        start_time = m.get('marketStartTime', '')
        
        if start_time:
            try:
                date_str = start_time[:10]  # YYYY-MM-DD
            except:
                continue
        else:
            continue
        
        runners = m.get('runners', [])
        for r in runners:
            selection_id = r.get('selectionId')
            runner_name = r.get('runnerName', '').upper()
            
            # Extract just the dog name (remove box number if present)
            if '. ' in runner_name:
                runner_name = runner_name.split('. ', 1)[1]
            
            key = (venue, date_str, runner_name)
            runner_lookup[key] = (market_id, selection_id)
    
    print(f"Built lookup with {len(runner_lookup)} runners", flush=True)
    
    # Connect to DB and find matching entries
    conn = sqlite3.connect(DB_PATH)
    
    # Get entries needing BSP for these dates
    print("\nLoading DB entries...", flush=True)
    query = """
    SELECT ge.EntryID, UPPER(t.TrackName), rm.MeetingDate, UPPER(g.GreyhoundName)
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE ge.BSP IS NULL
      AND rm.MeetingDate >= ?
    """
    
    db_entries = conn.execute(query, (from_date.strftime("%Y-%m-%d"),)).fetchall()
    print(f"Found {len(db_entries)} DB entries needing BSP", flush=True)
    
    # Match entries to Betfair runners
    matched = []
    for entry_id, track, date, dog_name in db_entries:
        key = (track, date, dog_name)
        if key in runner_lookup:
            market_id, selection_id = runner_lookup[key]
            matched.append((entry_id, market_id, selection_id))
    
    print(f"Matched {len(matched)} entries to Betfair data", flush=True)
    
    if not matched:
        print("\nNo matches found. Track names may differ between DB and Betfair.", flush=True)
        
        # Debug: show some samples
        print("\nSample DB entries:", flush=True)
        for e in db_entries[:5]:
            print(f"  {e[1]} - {e[2]} - {e[3]}", flush=True)
        
        print("\nSample Betfair runners:", flush=True)
        for k in list(runner_lookup.keys())[:5]:
            print(f"  {k}", flush=True)
        
        conn.close()
        return
    
    # Get BSP for matched markets
    print("\nFetching BSP for matched markets...", flush=True)
    
    # Group by market_id for efficient API calls
    market_entries = {}
    for entry_id, market_id, selection_id in matched:
        if market_id not in market_entries:
            market_entries[market_id] = []
        market_entries[market_id].append((entry_id, selection_id))
    
    all_updates = []
    processed = 0
    
    # Fetch in batches of 40 markets
    market_ids = list(market_entries.keys())
    for i in range(0, len(market_ids), 40):
        batch = market_ids[i:i+40]
        books = get_market_book_with_sp(batch)
        
        for book in books:
            market_id = book.get('marketId')
            if market_id not in market_entries:
                continue
                
            runners = book.get('runners', [])
            for runner in runners:
                selection_id = runner.get('selectionId')
                sp_data = runner.get('sp', {})
                bsp = sp_data.get('actualSP')
                
                if bsp and bsp > 0:
                    # Find matching entries
                    for entry_id, sel_id in market_entries[market_id]:
                        if sel_id == selection_id:
                            all_updates.append((bsp, entry_id))
        
        processed += len(batch)
        print(f"  Processed {processed}/{len(market_ids)} markets, {len(all_updates)} BSP found", flush=True)
    
    # Update database
    if all_updates:
        print(f"\nUpdating {len(all_updates)} entries...", flush=True)
        conn.executemany("UPDATE GreyhoundEntries SET BSP = ? WHERE EntryID = ?", all_updates)
        conn.commit()
        print("Done!", flush=True)
    else:
        print("\nNo BSP data found in market books.", flush=True)
    
    cursor = conn.execute("SELECT COUNT(*) FROM GreyhoundEntries WHERE BSP IS NOT NULL")
    final_bsp = cursor.fetchone()[0]
    print(f"\nFinal BSP coverage: {final_bsp:,}", flush=True)
    
    conn.close()

if __name__ == "__main__":
    main()
