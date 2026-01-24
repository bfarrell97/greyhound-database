import os
import sys
import logging
from datetime import datetime
# Setup path to include root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from scripts.predict_lay_strategy import run_daily_predictions

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'daily_workflow_{datetime.now().strftime("%Y%m%d")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        logging.info("Starting Daily Workflow...")
        
        # 1. Run Predictions
        logging.info("Running Prediction Pipeline...")
        candidates = run_daily_predictions()
        
        if not candidates:
            logging.warning("No Lay Candidates generated for today.")
            return

        # 2. Save Results
        df = pd.DataFrame(candidates)
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{datetime.now().strftime('%Y-%m-%d')}_lay_candidates.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logging.info(f"Successfully saved {len(df)} candidates to: {filepath}")
        
        # 3. Print Summary
        print("\n" + "="*50)
        print(f"DAILY SUMMARY: {datetime.now().strftime('%Y-%m-%d')}")
        print("="*50)
        print(df[['Track', 'Race', 'Dog', 'Margin', 'Odds']].to_string(index=False))
        print("="*50)
        
    except Exception as e:
        logging.error(f"Workflow Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
