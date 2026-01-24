import pandas as pd
import os

csv_path = 'live_bets.csv'
if not os.path.exists(csv_path):
    print("No CSV found.")
    exit()

df = pd.read_csv(csv_path)
initial_len = len(df)

# Filter out 2026-01-02
# Date format is YYYY-MM-DD
df = df[df['Date'] != '2026-01-02']

removed = initial_len - len(df)
print(f"Removed {removed} rows with date 2026-01-02.")

if removed > 0:
    df.to_csv(csv_path, index=False)
    print("Updated live_bets.csv")
else:
    print("No rows found to remove.")
