import betfair_data
import os
from datetime import datetime

FILE_PATH = "temp_debug_data/1.249872384.bz2"

def inspect():
    print(f"Inspecting {FILE_PATH}...")
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    count = 0
    for file_obj in betfair_data.Files([FILE_PATH]):
        # print(f"File: {file_obj.file_path}")
        for market in file_obj:
            print(f"Market: {market.market_id}")
            print(f"Venue: {market.venue}, Time: {market.market_time}")
            
            start_time = market.market_time
            
            for update in market:
                count += 1
                pt = update.publish_time
                time_diff = (start_time - pt).total_seconds() / 60.0
                
                # Count price updates
                price_updates = 0
                if update.runner_updates:
                    for runner in update.runner_updates:
                        if runner.best_available_to_back:
                            price_updates += 1
                
                print(f"Update {count}: PT={pt}, MinsOut={time_diff:.1f}, Prices={price_updates}")

if __name__ == "__main__":
    inspect()
