import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
SELECT
    ge.EntryID,
    g.GreyhoundName,
    t.TrackName,
    r.Distance,
    rm.MeetingDate,
    ge.FinishTime,
    ge.Split,
    ge.FinishTimeBenchmarkLengths,
    ge.SplitBenchmarkLengths
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE t.TrackName = 'The Gardens'
  AND (rm.MeetingDate LIKE '2025-07-11%' OR rm.MeetingDate LIKE '2025-08-01%')
LIMIT 15
""")

results = cursor.fetchall()

print('The Gardens races on 2025-07-11 and 2025-08-01:')
print(f'Found {len(results)} entries\n')

for r in results:
    ft = f'{r["FinishTime"]:.2f}' if r["FinishTime"] else "N/A"
    fb = f'{r["FinishTimeBenchmarkLengths"]:.2f}' if r["FinishTimeBenchmarkLengths"] else "N/A"
    sb = f'{r["SplitBenchmarkLengths"]:.2f}' if r["SplitBenchmarkLengths"] else "N/A"
    print(f'{r["GreyhoundName"]:20} {r["MeetingDate"][:10]} {r["Distance"]:3}m - '
          f'FinishTime:{ft:>6} FinishBench:{fb:>6} SplitBench:{sb:>6}')

conn.close()
