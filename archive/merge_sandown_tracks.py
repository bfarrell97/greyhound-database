"""
Merge Sandown Park (TrackID=332) into Sandown (SAP) (TrackID=903)
They are the same track - Sandown Park was renamed to Sandown (SAP)
"""
import sqlite3

conn = sqlite3.connect('greyhound_racing.db')
c = conn.cursor()

print("="*80)
print("MERGING SANDOWN PARK INTO SANDOWN (SAP)")
print("="*80)

# Check current state
c.execute('SELECT COUNT(*) FROM RaceMeetings WHERE TrackID = 332')
old_meetings = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM RaceMeetings WHERE TrackID = 903')
new_meetings = c.fetchone()[0]

print(f"\nBefore merge:")
print(f"  Sandown Park (TrackID=332): {old_meetings} meetings")
print(f"  Sandown (SAP) (TrackID=903): {new_meetings} meetings")

# Update all references from TrackID=332 to TrackID=903
print(f"\nMerging all Sandown Park data into Sandown (SAP)...")

# Update RaceMeetings table
c.execute('UPDATE RaceMeetings SET TrackID = 903 WHERE TrackID = 332')
updated_meetings = c.rowcount
print(f"  Updated {updated_meetings} race meetings")

# Check after merge
c.execute('SELECT COUNT(*) FROM RaceMeetings WHERE TrackID = 332')
old_after = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM RaceMeetings WHERE TrackID = 903')
new_after = c.fetchone()[0]

print(f"\nAfter merge:")
print(f"  Sandown Park (TrackID=332): {old_after} meetings")
print(f"  Sandown (SAP) (TrackID=903): {new_after} meetings")

# Commit changes
conn.commit()
print(f"\n[OK] Successfully merged Sandown Park into Sandown (SAP)")
print(f"[OK] TrackID=903 now has all historical data from both tracks")

conn.close()

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)
print("\nThe model can now use all Sandown historical data for predictions!")
