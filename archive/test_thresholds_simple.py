"""Test confidence thresholds"""
import sqlite3
import pandas as pd
import numpy as np
from greyhound_ml_model import GreyhoundMLModel

DB_PATH = 'greyhound_racing.db'
START_DATE = '2025-01-01'
END_DATE = '2025-11-30'
INITIAL_BANKROLL = 1000.0

print("="*80)
print("TESTING CONFIDENCE THRESHOLDS")
print("="*80)

ml_model = GreyhoundMLModel()
ml_model.load_model()

conn = sqlite3.connect(DB_PATH)
query = "SELECT ge.EntryID, ge.Position, ge.StartingPrice FROM GreyhoundEntries ge JOIN Races r ON ge.RaceID = r.RaceID JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ? AND ge.Position IS NOT NULL LIMIT 5"
df = pd.read_sql_query(query, conn, params=(START_DATE, END_DATE))
print(f"Test query returned {len(df)} rows")
conn.close()
