import pandas as pd
try:
    df = pd.read_csv('live_bets.csv')
    total = df['Profit'].fillna(0).sum()
    count = df['Profit'].count()
    print(f"Total Profit/Loss: ${total:,.2f}")
    print(f"Bets Settled: {count}")
    print(f"Total Rows: {len(df)}")
    print("-" * 20)
    print(df['Result'].value_counts())
except Exception as e:
    print(f"Error: {e}")
