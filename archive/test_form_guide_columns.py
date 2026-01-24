"""Test that form guide query returns all required columns including benchmark adjustments"""

import sqlite3

db_path = 'greyhound_racing.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get a sample greyhound with historical data
cursor.execute("""
    SELECT g.GreyhoundName
    FROM Greyhounds g
    JOIN GreyhoundEntries ge ON g.GreyhoundID = ge.GreyhoundID
    WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
    GROUP BY g.GreyhoundName
    HAVING COUNT(*) >= 5
    LIMIT 1
""")

result = cursor.fetchone()
if not result:
    print("No greyhounds found with benchmark data")
    conn.close()
    exit()

test_greyhound = result['GreyhoundName']
print(f"Testing form guide for: {test_greyhound}")
print("=" * 100)

# Run the same query as get_greyhound_form
cursor.execute("""
    SELECT
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        r.Grade,
        ge.Box as BoxNumber,
        ge.Position,
        ge.Margin,
        ge.FinishTime,
        ge.Split as FirstSectional,
        COALESCE(ge.InRun, ge.Form) as RunningPosition,
        ge.StartingPrice,
        tr.TrainerName,
        ge.SplitBenchmarkLengths as GFirstSecADJ,
        rm.MeetingSplitAvgBenchmarkLengths as MFirstSecADJ,
        ge.FinishTimeBenchmarkLengths as GOTADJ,
        rm.MeetingAvgBenchmarkLengths as MOTADJ
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    LEFT JOIN Trainers tr ON ge.TrainerID = tr.TrainerID
    WHERE g.GreyhoundName = ?
    ORDER BY rm.MeetingDate DESC, r.RaceNumber DESC
    LIMIT 5
""", (test_greyhound,))

races = cursor.fetchall()

print(f"\nShowing last 5 races for {test_greyhound}:")
print("-" * 100)
print(f"{'Date':<12} {'Track':<20} {'Dist':<5} {'Time':<7} {'Split':<7} {'G OT ADJ':<10} {'M OT ADJ':<10} {'G/M OT':<10}")
print("-" * 100)

for race in races:
    date = race['MeetingDate'] or '-'
    track = race['TrackName'][:19] or '-'
    dist = f"{race['Distance']}m" if race['Distance'] else '-'
    time = f"{race['FinishTime']:.2f}s" if race['FinishTime'] else '-'
    split = f"{race['FirstSectional']:.2f}s" if race['FirstSectional'] else '-'

    g_ot = race['GOTADJ']
    m_ot = race['MOTADJ']

    # Format without L suffix and without + for positive
    if g_ot is not None:
        g_ot_str = f"{g_ot:.2f}" if g_ot >= 0 else f"{g_ot:.2f}"
    else:
        g_ot_str = '-'

    if m_ot is not None:
        m_ot_str = f"{m_ot:.2f}" if m_ot >= 0 else f"{m_ot:.2f}"
    else:
        m_ot_str = '-'

    if g_ot is not None and m_ot is not None:
        gm_ot = g_ot - m_ot
        gm_ot_str = f"{gm_ot:.2f}" if gm_ot >= 0 else f"{gm_ot:.2f}"
    else:
        gm_ot_str = '-'

    print(f"{date:<12} {track:<20} {dist:<5} {time:<7} {split:<7} {g_ot_str:<10} {m_ot_str:<10} {gm_ot_str:<10}")

print("\n" + "=" * 100)
print("Column test complete! All benchmark columns are available.")
print("=" * 100)

# Show interpretation of one race
if races:
    race = races[0]
    print(f"\nInterpretation of most recent race:")
    print(f"  Dog: {test_greyhound}")
    print(f"  Date: {race['MeetingDate']}")
    print(f"  Track/Distance: {race['TrackName']} {race['Distance']}m")

    if race['GOTADJ'] is not None:
        if race['GOTADJ'] > 0:
            print(f"  G OT ADJ: {race['GOTADJ']:+.2f}L (ran {abs(race['GOTADJ']):.2f} lengths FASTER than benchmark)")
        else:
            print(f"  G OT ADJ: {race['GOTADJ']:+.2f}L (ran {abs(race['GOTADJ']):.2f} lengths SLOWER than benchmark)")

    if race['MOTADJ'] is not None:
        if race['MOTADJ'] > 0:
            print(f"  M OT ADJ: {race['MOTADJ']:+.2f}L (meeting was {abs(race['MOTADJ']):.2f} lengths FASTER than benchmark)")
        else:
            print(f"  M OT ADJ: {race['MOTADJ']:+.2f}L (meeting was {abs(race['MOTADJ']):.2f} lengths SLOWER than benchmark)")

    if race['GOTADJ'] is not None and race['MOTADJ'] is not None:
        gm_diff = race['GOTADJ'] - race['MOTADJ']
        print(f"  G/M OT ADJ: {gm_diff:+.2f}L (adjusted for track conditions)")
        if gm_diff > 0:
            print(f"    -> Dog ran {abs(gm_diff):.2f} lengths FASTER than meeting average")
        else:
            print(f"    -> Dog ran {abs(gm_diff):.2f} lengths SLOWER than meeting average")

conn.close()
