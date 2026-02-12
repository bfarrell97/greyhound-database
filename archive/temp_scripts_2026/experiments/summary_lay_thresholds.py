import pandas as pd
pd.set_option('display.float_format', '{:.3f}'.format)
df = pd.read_csv('outputs/grid_backtest_2025H2.csv')
# Overall by LAY_thresh
grp = df.groupby('LAY_thresh').agg(
    LAY_bets_total=('LAY_bets','sum'),
    LAY_pnl_total=('LAY_pnl','sum'),
    LAY_SR_avg=('LAY_SR','mean'),
    LAY_ROI_avg=('LAY_ROI','mean'),
    COMBINED_bets_total=('COMBINED_bets','sum'),
    COMBINED_pnl_total=('COMBINED_pnl','sum')
).reset_index()
# compute combined ROI per LAY
grp['COMBINED_ROI_pct'] = grp['COMBINED_pnl_total'] / grp['COMBINED_bets_total'] * 100
print('\n=== Overall by LAY threshold (Jul-Dec 2025) ===')
print(grp.to_string(index=False))

# Filter for BACK = 0.30
back03 = df[df['BACK_thresh']==0.3].copy()
back03 = back03.sort_values('LAY_thresh')
print('\n=== BACK = 0.30 rows ===')
print(back03[['LAY_thresh','BACK_bets','BACK_pnl','BACK_SR','BACK_ROI','LAY_bets','LAY_pnl','COMBINED_bets','COMBINED_pnl','COMBINED_ROI']].to_string(index=False))
