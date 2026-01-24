import pandas as pd
import numpy as np

strategies = [
    {'name': 'PIR + Pace Leader + $30k', 'file': 'results/backtest_longterm_leader.csv'},
    {'name': 'PIR + Pace Top 3 + $30k', 'file': 'results/backtest_longterm_top3.csv'}
]

output_file = 'weekly_report.md'

with open(output_file, 'w') as f:
    f.write("# 2025 Weekly Performance Breakdown\n")
    f.write("Period: 2025-01-01 to 2025-12-09\n\n")

    for strategy in strategies:
        csv_file = strategy['file']
        name = strategy['name']
        
        try:
            df = pd.read_csv(csv_file)
            df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
            
            # Filter for 2025
            df_2025 = df[df['MeetingDate'].dt.year == 2025].copy()
            
            if len(df_2025) == 0:
                f.write(f"## {name}\nNo bets found for 2025.\n\n")
                continue
                
            # Add Week Number
            df_2025['Week'] = df_2025['MeetingDate'].dt.isocalendar().week
            
            # Group by Week
            weekly = df_2025.groupby('Week').agg({
                'Profit': 'sum',
                'Stake': 'sum',
                'IsWinner': 'sum',
                'RaceKey': 'count'
            }).rename(columns={'RaceKey': 'Bets'})
            
            weekly['StrikeRate'] = (weekly['IsWinner'] / weekly['Bets']) * 100
            weekly['ROI'] = (weekly['Profit'] / weekly['Stake']) * 100
            weekly = weekly.sort_index()
            
            # Write to Markdown
            f.write(f"## Strategy: {name}\n")
            
            # Stats
            total_profit = weekly['Profit'].sum()
            total_roi = (total_profit / weekly['Stake'].sum()) * 100
            win_rate_week = (weekly['Profit'] > 0).sum() / len(weekly) * 100
            
            f.write(f"- **Total Bets:** {int(weekly['Bets'].sum()):,}\n")
            f.write(f"- **Total Profit:** {total_profit:+,.2f}u\n")
            f.write(f"- **Total ROI:** {total_roi:+.1f}%\n")
            f.write(f"- **Weekly Win Rate:** {win_rate_week:.0f}% ({int((weekly['Profit'] > 0).sum())}/{len(weekly)})\n\n")
            
            f.write("| Week | Bets | Wins | Strike Rate | Profit | ROI |\n")
            f.write("|:----:|:----:|:----:|:-----------:|:------:|:---:|\n")
            
            for week, row in weekly.iterrows():
                f.write(f"| {week} | {int(row['Bets'])} | {int(row['IsWinner'])} | {row['StrikeRate']:.1f}% | {row['Profit']:+.2f} | {row['ROI']:+.1f}% |\n")
            
            f.write("\n")

        except Exception as e:
            f.write(f"Error processing {name}: {e}\n")

print(f"Report written to {output_file}")
