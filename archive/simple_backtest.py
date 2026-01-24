"""
Simple backtest - just run predictions on November dates and check outcomes
"""
import sqlite3
import pandas as pd
from greyhound_ml_model import GreyhoundMLModel
from datetime import datetime

print("="*80)
print("SIMPLE NOVEMBER 2025 BACKTEST")
print("="*80)

# This approach: Just use the existing predict function on historical dates
# and compare to actual outcomes

m = GreyhoundMLModel()
m.load_model()

print("\nModel loaded successfully")

# Get list of November dates where we have results
conn = sqlite3.connect('greyhound_racing.db')
dates_query = """
SELECT DISTINCT rm.MeetingDate, COUNT(DISTINCT r.RaceID) as races
FROM RaceMeetings rm
JOIN Races r ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-11-01' AND rm.MeetingDate < '2025-12-01'
GROUP BY rm.MeetingDate
ORDER BY rm.MeetingDate
"""

dates_df = pd.read_sql_query(dates_query, conn)
print(f"\nFound {len(dates_df)} days with race results in November 2025")

# Sample just first few days to speed things up
sample_dates = dates_df['MeetingDate'].tolist()[:5]  # Just test first 5 days
print(f"Testing on {len(sample_dates)} sample days: {', '.join(sample_dates)}")

all_results = []

for test_date in sample_dates:
    print(f"\n  Testing {test_date}...")

    try:
        # Get actual results for this date
        results_query = """
        SELECT
            g.GreyhoundName,
            t.TrackName,
            r.RaceNumber,
            ge.Position,
            ge.StartingPrice
        FROM GreyhoundEntries ge
        JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate = ?
        AND ge.Position IS NOT NULL
        AND ge.Position NOT IN ('DNF', 'SCR')
        """

        actual_results = pd.read_sql_query(results_query, conn, params=(test_date,))
        actual_results['PositionNum'] = pd.to_numeric(actual_results['Position'], errors='coerce')
        actual_results['Won'] = (actual_results['PositionNum'] == 1).astype(int)

        print(f"    Found {len(actual_results)} entries, {actual_results['Won'].sum()} winners")

        # SKIP predictions for now - the issue is the model needs "upcoming" data
        # which doesn't exist for historical dates
        # Instead, let's just show what we would need to do

        print(f"    (Skipping predictions - would need to populate UpcomingBetting tables)")

    except Exception as e:
        print(f"    Error: {e}")

conn.close()

print("\n" + "="*80)
print("SIMPLIFIED VALIDATION APPROACH")
print("="*80)
print("""
ISSUE: The model's predict_upcoming_races() function requires data in the
UpcomingBettingRaces/UpcomingBettingRunners tables. Historical data is in
Races/GreyhoundEntries tables.

TWO OPTIONS TO VALIDATE:

OPTION 1: Forward-only validation (RECOMMENDED)
   - Start tracking predictions from TODAY forward
   - Save predictions to a log file each day
   - Compare to actual results the next day
   - Build up 2-4 weeks of real validation data
   - This is the SAFEST approach before betting money

OPTION 2: Backfill November data into UpcomingBetting tables
   - Copy November historical data into UpcomingBetting format
   - Run model predictions
   - Compare to actual results
   - More work but gives immediate validation

CURRENT STATUS:
Your model trained on: 2023-2024 data
Your model predicting: December 2025 (1 year in future)
Validation status: NONE - No testing on 2025 data yet

RECOMMENDATION:
Start paper trading TODAY:
1. Each evening, save tomorrow's predictions to a file
2. Next day, check actual results
3. After 14 days, calculate win rate and ROI
4. Only bet real money if win rate > 15% and ROI > 0%
""")
