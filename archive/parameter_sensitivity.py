import sqlite3
import pandas as pd
import numpy as np
import itertools

# Configuration
DB_PATH = 'greyhound_racing.db'
start_date = '2020-01-01'
end_date = '2025-12-09'

def load_data():
    print("Loading data for sensitivity analysis...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Fetch Races
    races_query = f"""
    SELECT 
        rm.MeetingID,
        rm.MeetingDate,
        ge.GreyhoundID,
        ge.Box,
        ge.StartingPrice as CurrentOdds,
        ge.CareerPrizeMoney,
        (CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as IsWinner,
        t.TrackName,
        r.RaceNumber
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '{start_date}' 
      AND rm.MeetingDate <= '{end_date}'
      AND ge.StartingPrice IS NOT NULL
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.Box IS NOT NULL
    """
    df = pd.read_sql_query(races_query, conn)
    
    # Cleaning
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce')
    df['CurrentOdds'] = pd.to_numeric(df['CurrentOdds'], errors='coerce')
    df['CareerPrizeMoney'] = df['CareerPrizeMoney'].astype(str).str.replace(r'[$,]', '', regex=True)
    df['CareerPrizeMoney'] = pd.to_numeric(df['CareerPrizeMoney'], errors='coerce').fillna(0)
    
    # 2. History & Metrics (Simplified for speed - reusing bulk logic)
    print("Loading history & metrics...")
    unique_dogs = df['GreyhoundID'].unique()
    dogs_str = ",".join(map(str, unique_dogs))
    
    hist_query = f"""
    SELECT 
        ge.GreyhoundID,
        rm.MeetingDate,
        ge.FirstSplitPosition,
        (ge.FinishTimeBenchmarkLengths + COALESCE(rm.MeetingAvgBenchmarkLengths, 0)) as TotalPace
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    WHERE ge.GreyhoundID IN ({dogs_str})
      AND rm.MeetingDate < '{end_date}'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    ORDER BY ge.GreyhoundID, rm.MeetingDate ASC
    """
    history_df = pd.read_sql_query(hist_query, conn)
    conn.close()
    
    # Calc Rolling
    history_df['MeetingDate'] = pd.to_datetime(history_df['MeetingDate'])
    
    split_hist = history_df.dropna(subset=['FirstSplitPosition']).copy()
    split_hist['HistAvgSplit'] = split_hist.groupby('GreyhoundID')['FirstSplitPosition'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    pace_hist = history_df.dropna(subset=['TotalPace']).copy()
    pace_hist['HistAvgPace'] = history_df.groupby('GreyhoundID')['TotalPace'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=5).mean()
    )
    
    # Merge
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df = df.merge(split_hist[['GreyhoundID', 'MeetingDate', 'HistAvgSplit']], on=['GreyhoundID', 'MeetingDate'], how='left')
    df = df.merge(pace_hist[['GreyhoundID', 'MeetingDate', 'HistAvgPace']], on=['GreyhoundID', 'MeetingDate'], how='left')
    
    df = df.dropna(subset=['HistAvgSplit', 'HistAvgPace']).copy()
    
    # Predictions
    box_adj = {1: -0.3, 2: -0.2, 3: -0.1, 4: 0, 5: 0, 6: 0.1, 7: 0.2, 8: 0.3}
    df['BoxAdj'] = df['Box'].map(box_adj).fillna(0)
    df['PredictedPIR'] = df['HistAvgSplit'] + df['BoxAdj']
    
    # Ranking
    df['RaceKey'] = df['MeetingID'].astype(str) + '_R' + df['RaceNumber'].astype(str)
    df['PredictedPIRRank'] = df.groupby('RaceKey')['PredictedPIR'].rank(method='min')
    df['PaceRank'] = df.groupby('RaceKey')['HistAvgPace'].rank(method='min', ascending=True)
    
    return df

def analyze_sensitivity(master_df):
    print("\nRunning Parameter Sensitivity Check...")
    print("="*80)
    print(f"{'PRIZE':<10} | {'ODDS MIN':<10} | {'STRATEGY':<10} | {'BETS':<8} | {'SR':<7} | {'ROI':<8} | {'NET PROFIT':<12}")
    print("-" * 80)
    
    # Parameters to test
    prize_levels = [0, 10000, 20000, 30000, 40000, 50000]
    odds_mins = [1.0, 1.5, 2.0]
    strategies = ['Leader', 'Top3'] # Leader = PaceRank 1, Top3 = PaceRank <= 3
    
    results = []
    
    for prize, odd_min, strat in itertools.product(prize_levels, odds_mins, strategies):
        # Filter
        df = master_df.copy()
        
        # Strategy Logic
        is_pir_leader = df['PredictedPIRRank'] == 1
        
        if strat == 'Leader':
            is_pace_qualifier = df['PaceRank'] == 1
        else:
            is_pace_qualifier = df['PaceRank'] <= 3
            
        has_money = df['CareerPrizeMoney'] >= prize
        in_odds = (df['CurrentOdds'] >= odd_min) & (df['CurrentOdds'] <= 30.0)
        
        bets = df[is_pir_leader & is_pace_qualifier & has_money & in_odds].copy()
        
        if len(bets) < 100:
            continue
            
        # Calc Stats (Flat Staking for comparable baseline)
        bets['FlatProfit'] = np.where(bets['IsWinner']==1, bets['CurrentOdds']-1, -1)
        
        n_bets = len(bets)
        sr = (bets['IsWinner'].sum() / n_bets) * 100
        roi = (bets['FlatProfit'].sum() / n_bets) * 100
        
        # Net Profit (5% comm)
        gross_win = bets[bets['IsWinner']==1]['FlatProfit'].sum()
        loss = bets[bets['IsWinner']==0]['FlatProfit'].sum()
        net_profit = (gross_win * 0.95) + loss
        
        print(f"${prize/1000:<3.0f}k      | {odd_min:<10} | {strat:<10} | {n_bets:<8} | {sr:<6.1f}% | {roi:<7.1f}% | {net_profit:+.0f}u")
        results.append({
            'Prize': prize,
            'OddsMin': odd_min,
            'Strategy': strat,
            'Bets': n_bets,
            'ROI': roi,
            'NetProfit': net_profit
        })
        
    print("="*80)
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_data()
    results = analyze_sensitivity(df)
    results.to_csv('sensitivity_results.csv', index=False)
