"""
Backtest V44/V45 at different thresholds - Last 6 months
NOTE: This is IN-SAMPLE data (models trained on full DB), results will be optimistic.
"""
import sqlite3
import pandas as pd
import sys
sys.path.insert(0, '.')

from scripts.predict_v44_prod import MarketAlphaEngine

print("="*70)
print("V44/V45 THRESHOLD BACKTEST (Last 6 Months - IN-SAMPLE)")
print("="*70)

print("\nLoading Engine...")
engine = MarketAlphaEngine(db_path='greyhound_racing.db')

# Get 6 months of data
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT 
    ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
    ge.Position, ge.BSP, ge.Price5Min, ge.Weight, ge.TrainerID,
    ge.Split, ge.FinishTime, ge.Margin,
    r.Distance, r.Grade, r.RaceNumber, t.TrackName, rm.MeetingDate,
    g.DateWhelped, g.GreyhoundName as Dog, r.RaceTime
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate >= date('now', '-180 days')
AND ge.BSP > 0
AND ge.Price5Min > 0
AND ge.Price5Min < 30
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df)} runners from last 6 months")

# Run predictions
print("Running predictions (this may take a minute)...")
results = engine.predict(df, use_cache=True)

def evaluate_threshold(df, back_thresh, lay_thresh, label):
    """Evaluate performance at given thresholds"""
    print(f"\n--- {label} ---")
    print(f"BACK >= {back_thresh}, LAY >= {lay_thresh}")
    
    # BACK signals
    back = df[df['Steam_Prob'] >= back_thresh].copy()
    if len(back) > 0:
        back['Beat_BSP'] = back['Price5Min'] > back['BSP']
        back['Won'] = back['Position'] == 1
        back_beat = back['Beat_BSP'].mean() * 100
        back_win = back['Won'].mean() * 100
        # Simple ROI: Assume $1 bet at Price5Min, return BSP if win
        back['Profit'] = back.apply(lambda x: x['BSP'] - 1 if x['Position'] == 1 else -1, axis=1)
        back_roi = back['Profit'].sum() / len(back) * 100
        print(f"  BACK: {len(back):5d} signals | Beat BSP: {back_beat:5.1f}% | Win: {back_win:5.1f}% | ROI: {back_roi:+5.1f}%")
    else:
        print(f"  BACK: 0 signals")
    
    # LAY signals
    lay = df[df['Drift_Prob'] >= lay_thresh].copy()
    if len(lay) > 0:
        lay['Beat_BSP'] = lay['Price5Min'] < lay['BSP']
        lay['Won'] = lay['Position'] != 1  # LAY wins if dog loses
        lay_beat = lay['Beat_BSP'].mean() * 100
        lay_win = lay['Won'].mean() * 100
        # LAY ROI: Win = +1 (stake), Lose = -(BSP-1) liability
        lay['Profit'] = lay.apply(lambda x: 1 if x['Position'] != 1 else -(x['BSP'] - 1), axis=1)
        lay_roi = lay['Profit'].sum() / len(lay) * 100
        print(f"  LAY:  {len(lay):5d} signals | Beat BSP: {lay_beat:5.1f}% | Win: {lay_win:5.1f}% | ROI: {lay_roi:+5.1f}%")
    else:
        print(f"  LAY:  0 signals")

# Test different thresholds
evaluate_threshold(results, 0.35, 0.60, "CURRENT THRESHOLDS")
evaluate_threshold(results, 0.30, 0.55, "LOWERED BY 0.05")
evaluate_threshold(results, 0.25, 0.50, "LOWERED BY 0.10")
evaluate_threshold(results, 0.20, 0.45, "AGGRESSIVE (0.20/0.45)")

print("\n" + "="*70)
print("REMINDER: This is IN-SAMPLE. Live performance will differ.")
print("="*70)
