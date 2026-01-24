"""Compare Price5Min to BSP"""
import sqlite3
c = sqlite3.connect('greyhound_racing.db')

# Compare Price5Min to BSP where both exist
print("="*60)
print("PRICE5MIN vs BSP COMPARISON")
print("="*60)

query = """
SELECT COUNT(*), AVG(Price5Min/BSP), MIN(Price5Min/BSP), MAX(Price5Min/BSP)
FROM GreyhoundEntries
WHERE Price5Min IS NOT NULL AND BSP IS NOT NULL AND BSP > 0
"""
row = c.execute(query).fetchone()
print(f"Entries with both: {row[0]:,}")
print(f"Avg ratio (P5/BSP): {row[1]:.4f}")
print(f"This means prices at 5min are {(row[1]-1)*100:+.2f}% higher than BSP on average")

# Break down by BSP range
print("\nBy BSP range:")
print("-"*60)
ranges = [
    (1, 2.25, "Lay range (<$2.25)"),
    (2.25, 3, "Short ($2.25-$3)"),
    (3, 8, "Back range ($3-$8)"),
    (8, 20, "Mid ($8-$20)"),
    (20, 100, "Long ($20-$100)")
]
for lo, hi, label in ranges:
    q = f"""
    SELECT COUNT(*), AVG(Price5Min/BSP)
    FROM GreyhoundEntries
    WHERE Price5Min IS NOT NULL AND BSP IS NOT NULL AND BSP >= {lo} AND BSP < {hi}
    """
    r = c.execute(q).fetchone()
    if r[0] > 0:
        edge = (r[1] - 1) * 100
        print(f"{label:25} {r[0]:>8,} entries, ratio {r[1]:.4f} (back edge: {edge:+.2f}%)")

print("\nNOTE: Ratio > 1 means 5min price is HIGHER than BSP (good for backing)")
print("      Ratio < 1 means 5min price is LOWER than BSP (good for laying)")
