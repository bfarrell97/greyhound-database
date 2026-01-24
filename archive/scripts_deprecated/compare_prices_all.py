"""
Database-Wide Price Analysis: Price5Min vs BSP ($3-$8 Bracket)
"""
import sqlite3
import pandas as pd
import numpy as np

print("="*60)
print("DATABASE WIDE PRICE ANALYSIS ($3-$8 Bracket)")
print("Comparing BSP vs Price5Min for all runners")
print("="*60)

conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT BSP, Price5Min, Position, MeetingDate
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE ge.BSP BETWEEN 3 AND 8
  AND ge.Price5Min IS NOT NULL
  AND ge.Position IS NOT NULL 
  AND ge.Position NOT IN ('SCR', 'DNF', '')
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['BSP'] = pd.to_numeric(df['BSP'], errors='coerce')
df['Price5Min'] = pd.to_numeric(df['Price5Min'], errors='coerce')
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['Won'] = (df['Position'] == 1).astype(int)
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])

print(f"Loaded {len(df):,} runners in $3-$8 bracket with valid Price5Min")

if len(df) > 0:
    # Basic Stats
    avg_bsp = df['BSP'].mean()
    avg_p5 = df['Price5Min'].mean()
    diff = (avg_p5 - avg_bsp) / avg_bsp * 100
    
    print(f"\nAverage Prices:")
    print(f"  BSP:       ${avg_bsp:.3f}")
    print(f"  Price5Min: ${avg_p5:.3f}")
    print(f"  Difference: {diff:+.2f}%")
    
    # Frequency
    p5_higher = sum(df['Price5Min'] > df['BSP'])
    bsp_higher = sum(df['BSP'] > df['Price5Min'])
    equal = sum(df['BSP'] == df['Price5Min'])
    
    print(f"\nPrice Movement Direction:")
    print(f"  Drifters (BSP > Price5Min): {bsp_higher:,} ({bsp_higher/len(df)*100:.1f}%)")
    print(f"  Steamers (Price5Min > BSP): {p5_higher:,} ({p5_higher/len(df)*100:.1f}%)")
    print(f"  No Change:                  {equal:,} ({equal/len(df)*100:.1f}%)")
    
    # Profitability (Flat Stakes)
    winners = df[df['Won'] == 1]
    sr = len(winners) / len(df) * 100
    
    profit_bsp = winners['BSP'].sum() - len(df)
    roi_bsp = profit_bsp / len(df) * 100
    
    profit_p5 = winners['Price5Min'].sum() - len(df)
    roi_p5 = profit_p5 / len(df) * 100
    
    print(f"\nProfitability (Flat Stakes, No Commission):")
    print(f"  Win Rate: {sr:.2f}%")
    print(f"  ROI @ BSP:       {roi_bsp:+.2f}%")
    print(f"  ROI @ Price5Min: {roi_p5:+.2f}%")
    
    if roi_bsp > roi_p5:
        print(f"\n-> BSP is better by {roi_bsp - roi_p5:.2f}% ROI points")
    else:
        print(f"\n-> Price5Min is better by {roi_p5 - roi_bsp:.2f}% ROI points")

    # Monthly Trend
    print(f"\nMonthly Trend (BSP vs Price5Min Diff):")
    df['Month'] = df['MeetingDate'].dt.to_period('M')
    monthly = df.groupby('Month').apply(lambda x: (x['Price5Min'].mean() - x['BSP'].mean()) / x['BSP'].mean() * 100)
    print(monthly.tail(12))

else:
    print("No data found matching criteria.")

print("="*60)
