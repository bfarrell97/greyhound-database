import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('greyhound_racing.db')
today = datetime.now().strftime('%Y-%m-%d')

# Check LiveBets count
q = "SELECT COUNT(*) as cnt FROM LiveBets WHERE BetDate = ?"
r = pd.read_sql_query(q, conn, params=[today])
print(f"Live Bets Today: {r.iloc[0]['cnt']}")

# Show last 5 bets
q2 = "SELECT * FROM LiveBets WHERE BetDate = ? ORDER BY ROWID DESC LIMIT 5"
r2 = pd.read_sql_query(q2, conn, params=[today])
print(r2)

# Check signals generated in last few hours
q3 = """
SELECT ge.EntryID, t.TrackName, r.RaceNumber, r.RaceTime, g.GreyhoundName, ge.Price5Min
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate = ?
AND ge.Price5Min IS NOT NULL
AND ge.Price5Min > 0
AND ge.Price5Min < 30
ORDER BY r.RaceTime DESC
LIMIT 20
"""
r3 = pd.read_sql_query(q3, conn, params=[today])
print(f"\nRecent Runners with Price5Min < $30 ({len(r3)} shown):")
print(r3)

conn.close()
