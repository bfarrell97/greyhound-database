import os
import csv
import glob
import logging
import bz2
import gzip
import betfairlightweight
from betfairlightweight import StreamListener
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data/bsp"
OUTPUT_FILE = "data/historical_bsp_parsed.csv"

# Function to open files transparently
def open_file(path):
    if path.endswith('.bz2'):
        return bz2.open(path, 'rt', encoding='utf-8')
    elif path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    else:
        return open(path, 'r', encoding='utf-8')

def process_file(file_path):
    """
    Process a single Betfair historical file and return list of extracted rows.
    """
    rows = []
    
    # We use a dummy client since we don't need to authenticate to process files
    # (Checking if auth is needed for historical stream... usually no)
    
    # Actually betfairlightweight needs a client object usually.
    # User example: trading = betfairlightweight.APIClient("username", "password")
    trading = betfairlightweight.APIClient("dummy", "dummy")
    
    listener = StreamListener(max_latency=None)
    
    # NOTE: The library might expect smart_open or a real path string if we pass file_path directly.
    # create_historical_generator_stream usually handles the opening if given a string.
    # But if smart_open is missing, we might need to pass a file object or monkey patch?
    # Let's try passing the path first, but catch the ImportError if it happens inside the lib.
    
    try:
        stream = trading.streaming.create_historical_generator_stream(
            file_path=file_path,
            listener=listener,
        )
        gen = stream.get_generator()
        
        for market_books in gen():
            for market_book in market_books:
                # We only care about CLOSED markets to get the final BSP and Result
                # BUT the stream emits updates. We need to capture the Final state.
                # Actually, Historic Stream plays back the whole market. 
                # We can just look for the final update?
                # Or accumulate state?
                # The 'market_book' object returned by listener IS the accumulated state (it's a cache).
                
                # We want to extract data once the market is CLOSED (Settled).
                
                if market_book.status == "CLOSED":
                    market_def = market_book.market_definition
                    
                    # Basic checks
                    if not market_def: 
                        continue
                        
                    # Filter for Greyhounds if needed, but we assume input files are correct
                    # event_type_id 4339 = Greyhounds
                    
                    market_id = market_book.market_id
                    event_date = market_def.market_time
                    venue = market_def.venue_name or market_def.event_name # Sometimes one is missing
                    
                    for runner in market_book.runners:
                        selection_id = runner.selection_id
                        
                        # Get runner definition (name, sort_priority for Box)
                        runner_def = next((r for r in market_def.runners if r.selection_id == selection_id), None)
                        if not runner_def:
                            continue
                            
                        dog_name = runner_def.name
                        status = runner.status # WINNER / LOSER
                        
                        # BSP
                        bsp = runner.sp.actual_sp if runner.sp else None
                        
                        rows.append({
                            "MarketID": market_id,
                            "Date": event_date,
                            "Track": venue,
                            "DogName": dog_name,
                            "SelectionID": selection_id,
                            "Status": status,
                            "BSP": bsp,
                            "SortPriority": runner_def.sort_priority 
                        })
                        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        
    return rows

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist.")
        return

    # Find files (bz2, gz, or json)
    files = glob.glob(os.path.join(DATA_DIR, "*"))
    print(f"Found {len(files)} files in {DATA_DIR}")
    
    all_rows = []
    
    for i, f in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {f}")
        file_rows = process_file(f)
        all_rows.extend(file_rows)
        
    if not all_rows:
        print("No data parsed.")
        return
        
    # Write to CSV
    keys = ["MarketID", "Date", "Track", "DogName", "SelectionID", "Status", "BSP", "SortPriority"]
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"Successfully processed {len(all_rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
