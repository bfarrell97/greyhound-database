# explore_hybrid_staking.py
"""
Simulates various staking strategies for the Hybrid V28/V30 model.
Updated with realistic constraints and clearer metrics.

Filters: Value Threshold = 0.75, Price Cap = $8.
Initial Bankroll: $1,000.
Min Bet: $1.
Max Bet: $500 (Market Liquidity Constraint).
"""

import sqlite3
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from autogluon.tabular import TabularPredictor

# ---------- 1. DATA PREPARATION ----------
def load_and_prep_data():
    print("Loading data (2023-2025)...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT
        ge.EntryID, ge.RaceID as FastTrack_RaceId, ge.GreyhoundID as FastTrack_DogId,
        ge.Box, ge.Weight, ge.Position as Place, ge.BSP as StartPrice, ge.PrizeMoney as Prizemoney,
        ge.FinishTime as RunTime, ge.Split as SplitMargin,
        r.Distance, t.TrackName as Track, rm.MeetingDate as date_dt,
        ge.Price2Hr, ge.Price60Min, ge.Price30Min, ge.Price15Min, ge.Price10Min, ge.Price5Min, ge.Price2Min, ge.Price1Min
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2023-01-01' AND rm.MeetingDate < '2026-01-01'
      AND ge.Position IS NOT NULL AND ge.Position NOT IN ('SCR', 'DNF', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    for col in ['Place', 'RunTime', 'SplitMargin', 'StartPrice', 'Prizemoney', 'Distance', 'Box']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[df['Place'].notna() & (df['Box'] > 0)]
    df['win'] = (df['Place'] == 1).astype(int)
    
    # Simple features for inference
    df['StartPrice_probability'] = (1 / df['StartPrice']).fillna(0)
    df['StartPrice_probability'] = df.groupby('FastTrack_RaceId')['StartPrice_probability'].transform(lambda x: x / x.sum())
    
    df['Prizemoney_norm'] = np.log10(df['Prizemoney'] + 1) / 12
    df['Place_inv'] = (1 / df['Place']).fillna(0)
    df['Place_log'] = np.log10(df['Place'] + 1).fillna(0)
    df['BSP_log'] = np.log(df['StartPrice'].clip(lower=1.01)).fillna(0)
    
    # Track stats
    win_df = df[df['win'] == 1]
    median_win_time = win_df[win_df['RunTime'] > 0].groupby(['Track', 'Distance'])['RunTime'].median().reset_index()
    median_win_time.columns = ['Track', 'Distance', 'RunTime_median']
    median_win_split = win_df[win_df['SplitMargin'] > 0].groupby(['Track', 'Distance'])['SplitMargin'].median().reset_index()
    median_win_split.columns = ['Track', 'Distance', 'SplitMargin_median']
    
    median_win_time['speed_index'] = median_win_time['RunTime_median'] / median_win_time['Distance']
    median_win_time['speed_index'] = MinMaxScaler().fit_transform(median_win_time[['speed_index']])
    
    df = df.merge(median_win_time[['Track', 'Distance', 'RunTime_median', 'speed_index']], on=['Track', 'Distance'], how='left')
    df = df.merge(median_win_split, on=['Track', 'Distance'], how='left')
    
    df['RunTime_norm'] = (df['RunTime_median'] / df['RunTime']).clip(0.9, 1.1)
    df['RunTime_norm'] = MinMaxScaler().fit_transform(df[['RunTime_norm']])
    
    df['SplitMargin_norm'] = (df['SplitMargin_median'] / df['SplitMargin']).clip(0.9, 1.1)
    df['SplitMargin_norm'] = MinMaxScaler().fit_transform(df[['SplitMargin_norm']])
    
    box_win = df.groupby(['Track', 'Distance', 'Box'])['win'].mean().reset_index()
    box_win.columns = ['Track', 'Distance', 'Box', 'box_win_percent']
    df = df.merge(box_win, on=['Track', 'Distance', 'Box'], how='left')

    price_order = ['Price2Hr', 'Price60Min', 'Price30Min', 'Price15Min', 'Price10Min', 'Price5Min', 'Price2Min', 'Price1Min']
    for col in price_order:
        if col not in df.columns: df[col] = np.nan
    df['EarliestPrice'] = df[price_order].bfill(axis=1).iloc[:, 0]
    df['EarliestPrice'].fillna(df['StartPrice'], inplace=True)

    features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm', 'BSP_log']
    aggregates = ['min', 'max', 'mean', 'median', 'std']
    rolling_windows = ['28D', '91D', '365D']
    
    dataset = df.copy().set_index(['FastTrack_DogId', 'date_dt']).sort_index()
    feature_cols = ['speed_index', 'box_win_percent']
    
    print("Calculating rolling features...")
    for w in rolling_windows:
        rolling_res = (
            dataset.reset_index(level=0)
            .groupby('FastTrack_DogId')[features]
            .rolling(w)
            .agg(aggregates)
            .groupby(level=0)
            .shift(1)
        )
        agg_cols = [f"{f}_{a}_{w}" for f, a in itertools.product(features, aggregates)]
        dataset[agg_cols] = rolling_res
        feature_cols.extend(agg_cols)
        
    dataset.fillna(0, inplace=True)
    model_df = dataset.reset_index()
    feature_cols = list(set(feature_cols))
    
    return model_df, feature_cols

model_df, feature_cols = load_and_prep_data()
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'win', 'StartPrice', 'EarliestPrice', 'Distance'] + feature_cols]

# ---------- 2. MODELS & PREDICTIONS ----------
print("Loading models and predicting...")
predictor_v28 = TabularPredictor.load('models/autogluon_v28_tutorial')
predictor_v30 = TabularPredictor.load('models/autogluon_v30_bsp')

prob_v28 = predictor_v28.predict_proba(model_df)
prob_v30 = predictor_v30.predict_proba(model_df)
model_df['prob_model'] = (prob_v28[1] + prob_v30[1]) / 2
model_df['prob_model'] = model_df.groupby('FastTrack_RaceId')['prob_model'].transform(lambda x: x / x.sum())

# ---------- 3. IDENTIFY BETS ----------
VT = 0.75
CAP = 8
model_df['RatedPrice'] = 1 / model_df['prob_model']
bets = model_df[(model_df['EarliestPrice'] > model_df['RatedPrice'] * (1 + VT)) &
                (model_df['EarliestPrice'] <= CAP)].copy()
bets = bets.sort_values('date_dt')
bets = bets[bets['date_dt'] >= '2024-01-01']
print(f"Identified {len(bets)} bets using VT={VT}, Cap=${CAP} (2024-2025)")

# ---------- 4. SIMULATION ----------
INITIAL_BANK = 1000
MIN_BET = 1
MAX_BET = 500  # Market liquidity cap

def run_simulation(strategy_name, bet_logic_func):
    bankroll = INITIAL_BANK
    history = []
    total_staked = 0
    total_profit = 0
    
    for _, row in bets.iterrows():
        raw_stake = bet_logic_func(row, bankroll)
        
        # Apply Logic Limits
        stake = min(raw_stake, bankroll) # Can't bet more than bank
        stake = min(stake, MAX_BET)      # Market cap
        
        if stake < MIN_BET:
            history.append(bankroll)
            continue
            
        total_staked += stake
        
        if row['win'] == 1:
            # 10% commission on Net Profit
            # Profit = Revenue - Stake
            # Revenue = Stake * Odds
            # Gross Profit = Stake * (Odds - 1)
            # Net Profit = Gross Profit * 0.9
            profit = stake * (row['EarliestPrice'] - 1) * 0.9
        else:
            profit = -stake
            
        total_profit += profit
        bankroll += profit
        history.append(bankroll)
        
        if bankroll < MIN_BET: # Effectively busted
            bankroll = 0
            break
            
    history = np.array(history)
    high_water_mark = np.maximum.accumulate(history)
    drawdowns = (high_water_mark - history) / high_water_mark * 100
    
    yield_percent = (total_profit / total_staked * 100) if total_staked > 0 else 0
    bank_growth = ((bankroll - INITIAL_BANK) / INITIAL_BANK * 100)
    
    return {
        'Strategy': strategy_name,
        'Final Bank': f"${bankroll:,.0f}",
        'Net Profit': f"${total_profit:,.0f}",
        'Turnover': f"${total_staked:,.0f}",
        'Yield %': f"{yield_percent:.2f}%",
        'Bank Grow %': f"{bank_growth:,.0f}%",
        'Max DD %': f"{np.max(drawdowns):.2f}%" if len(drawdowns) > 0 else "0%",
        'Status': "Active" if bankroll > 10 else "Busted"
    }

# Staking Functions
def flat_10(row, bank): return 10
def pct_1(row, bank): return bank * 0.01
def pct_2(row, bank): return bank * 0.02
def pct_3(row, bank): return bank * 0.03

def kelly(row, bank, fraction=1.0):
    b = row['EarliestPrice'] - 1
    p = row['prob_model']
    q = 1 - p
    f = (b * p - q) / b
    return bank * (f * fraction) if f > 0 else 0

def kelly_sixteenth(row, bank): return kelly(row, bank, 0.0625)
def kelly_eighth(row, bank): return kelly(row, bank, 0.125)
def kelly_quarter(row, bank): return kelly(row, bank, 0.25)
def kelly_half(row, bank): return kelly(row, bank, 0.5)

def target_profit_10(row, bank):
    target = 10
    return target / (row['EarliestPrice'] - 1)

experiments = [
    ('Flat $10', flat_10),
    ('Percentage 1%', pct_1),
    ('Percentage 2%', pct_2),
    ('Percentage 3%', pct_3),
    ('Kelly 16th', kelly_sixteenth),
    ('Kelly Eighth', kelly_eighth),
    ('Kelly Quarter', kelly_quarter),
    ('Kelly Half', kelly_half),
    ('Target Profit $10', target_profit_10)
]

print("\n=== Staking Strategy Sim (Max Stake $500) ===")
results = []
for name, func in experiments:
    res = run_simulation(name, func)
    results.append(res)

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
