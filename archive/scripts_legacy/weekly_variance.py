import pandas as pd
import random

df = pd.read_csv('backtest_longterm_leader.csv')
df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Week'] = df['MeetingDate'].dt.to_period('W').astype(str)

# Group by week
weekly = df.groupby('Week').agg({
    'IsWinner': ['sum', 'count'],
    'FlatProfit': 'sum'
}).reset_index()
weekly.columns = ['Week', 'Wins', 'Bets', 'Profit']
weekly['ROI'] = (weekly['Profit'] / weekly['Bets'] * 100).round(1)
weekly['StrikeRate'] = (weekly['Wins'] / weekly['Bets'] * 100).round(1)

# Sample 20 random weeks
sample = weekly.sample(n=min(20, len(weekly)), random_state=42).sort_values('Week')

print('='*70)
print('RANDOM WEEKS SAMPLE - PIR + Pace Leader Strategy')
print('='*70)
print(f"{'Week':<15} | {'Bets':>5} | {'Wins':>5} | {'Strike%':>8} | {'Profit':>8} | {'ROI%':>8}")
print('-'*70)

total_bets = 0
total_profit = 0
total_wins = 0
winning_weeks = 0
losing_weeks = 0

for _, row in sample.iterrows():
    print(f"{row['Week']:<15} | {int(row['Bets']):>5} | {int(row['Wins']):>5} | {row['StrikeRate']:>7.1f}% | {row['Profit']:>+8.1f} | {row['ROI']:>+7.1f}%")
    total_bets += row['Bets']
    total_profit += row['Profit']
    total_wins += row['Wins']
    if row['Profit'] > 0: winning_weeks += 1
    else: losing_weeks += 1

print('-'*70)
print(f"{'TOTAL':<15} | {int(total_bets):>5} | {int(total_wins):>5} | {total_wins/total_bets*100:>7.1f}% | {total_profit:>+8.1f} | {total_profit/total_bets*100:>+7.1f}%")
print()
print(f'Winning Weeks: {winning_weeks} ({winning_weeks/(winning_weeks+losing_weeks)*100:.0f}%)')
print(f'Losing Weeks: {losing_weeks} ({losing_weeks/(winning_weeks+losing_weeks)*100:.0f}%)')

# Also write to file
with open('weekly_variance_report.txt', 'w', encoding='utf-8') as f:
    f.write('='*70 + '\n')
    f.write('RANDOM WEEKS SAMPLE - PIR + Pace Leader Strategy\n')
    f.write('='*70 + '\n')
    f.write(f"{'Week':<15} | {'Bets':>5} | {'Wins':>5} | {'Strike%':>8} | {'Profit':>8} | {'ROI%':>8}\n")
    f.write('-'*70 + '\n')
    for _, row in sample.iterrows():
        f.write(f"{row['Week']:<15} | {int(row['Bets']):>5} | {int(row['Wins']):>5} | {row['StrikeRate']:>7.1f}% | {row['Profit']:>+8.1f} | {row['ROI']:>+7.1f}%\n")
    f.write('-'*70 + '\n')
    f.write(f"{'TOTAL':<15} | {int(total_bets):>5} | {int(total_wins):>5} | {total_wins/total_bets*100:>7.1f}% | {total_profit:>+8.1f} | {total_profit/total_bets*100:>+7.1f}%\n")
    f.write(f'\nWinning Weeks: {winning_weeks} ({winning_weeks/(winning_weeks+losing_weeks)*100:.0f}%)\n')
    f.write(f'Losing Weeks: {losing_weeks} ({losing_weeks/(winning_weeks+losing_weeks)*100:.0f}%)\n')
print('Report saved to weekly_variance_report.txt')
