"""View benchmarks with split times"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
    SELECT * FROM Benchmarks
    ORDER BY TrackName, Distance
    LIMIT 20
""")

benchmarks = cursor.fetchall()

print("=" * 100)
print("BENCHMARKS WITH SPLIT TIMES (First 20)")
print("=" * 100)
print(f"\n{'Track':<25} {'Dist':<6} {'Avg Time':<10} {'Samples':<8} {'Avg Split':<10} {'Split N':<8}")
print("-" * 100)

for b in benchmarks:
    track = b['TrackName'][:24]
    dist = f"{b['Distance']}m"
    avg_time = f"{b['AvgTime']:.2f}s"
    samples = b['SampleSize']

    if b['AvgSplit']:
        avg_split = f"{b['AvgSplit']:.2f}s"
        split_n = b['SplitSampleSize'] or 0
    else:
        avg_split = "-"
        split_n = "-"

    print(f"{track:<25} {dist:<6} {avg_time:<10} {samples:<8} {avg_split:<10} {split_n:<8}")

conn.close()

print("\n" + "=" * 100)
print("Split times are now being tracked for benchmarks!")
print("Runners without split times are excluded from split benchmark calculations.")
print("=" * 100)
