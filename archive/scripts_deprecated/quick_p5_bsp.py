"""Compare Price5Min vs BSP at $3-$8 range"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')
q = """
SELECT AVG(Price5Min), AVG(BSP), AVG(Price5Min/BSP), COUNT(*)
FROM GreyhoundEntries
WHERE Price5Min IS NOT NULL AND BSP IS NOT NULL 
  AND BSP >= 3 AND BSP <= 8
"""
p5_avg, bsp_avg, ratio, cnt = c.execute(q).fetchone()
edge = (ratio - 1) * 100
print(f'Entries with both P5 and BSP at BSP $3-$8: {cnt:,}')
print(f'Avg Price5Min: ${p5_avg:.2f}')
print(f'Avg BSP: ${bsp_avg:.2f}')
print(f'Avg Ratio (P5/BSP): {ratio:.4f}')
print(f'Edge: {edge:+.2f}% (positive = P5 is higher = better for backing)')
