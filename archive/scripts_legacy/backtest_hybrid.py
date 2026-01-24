"""
Backtest Hybrid Strategy
Combine:
1. Regression Model (Time/Split) -> Identifies "Fastest Dog"
2. Probability Model (Win %) -> Identifies "Likely Winner" (Context)
Hypothesis: Betting on the Fastest Dog is only profitable when the Probability Model CONFIRMS it has a good chance (filters out fast dogs in bad boxes).
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

DB_PATH = 'greyhound_racing.db'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
PROB_MODEL_PATH = 'models/prob_xgb_model.pkl'

def load_data_and_features():
    # Load 2024-2025 Data
    conn = sqlite3.connect(DB_PATH)
    print("Loading Hybrid Test Data...")
    
    # Combined query
    query = """
    SELECT
        ge.GreyhoundID,
        g.GreyhoundName,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.Split,
        ge.FinishTime,
        ge.Position,
        ge.StartingPrice,
        COALESCE(ge.PrizeMoney, 0) as PrizeMoney
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2024-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    
    def parse_price(x):
        try:
            if not x or x is None: return np.nan
            x = str(x).replace('$', '').strip()
            if 'F' in x: x = x.replace('F', '')
            return float(x)
        except:
            return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # --- FEATURES ---
    print("Calculating Features...")
    
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Must have history
    df = df.dropna(subset=['s_Roll5', 'p_Roll5', 'Odds']).copy()
    
    return df

def run_hybrid_backtest(df):
    print("Loading Models & Predicting...")
    
    # Load Models
    with open(PIR_MODEL_PATH, 'rb') as f: pir_model = pickle.load(f)
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    with open(PROB_MODEL_PATH, 'rb') as f: prob_model = pickle.load(f)
    
    # Feature Sets
    # Regression: ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    cols_split = ['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']
    cols_pace = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    # Prob: ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    cols_prob = ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    
    X_split = df[cols_split].copy()
    X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    X_pace = df[cols_pace].copy()
    X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    
    X_prob = df[cols_prob].copy()
    
    # Predictions
    df['PredNormSplit'] = pir_model.predict(X_split)
    df['PredSplit'] = df['PredNormSplit'] + df['TrackDistMedianSplit']
    
    df['PredNormPace'] = pace_model.predict(X_pace)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    df['WinProb'] = prob_model.predict_proba(X_prob)[:, 1]
    
    # RANKING
    print("Simulating Strategies...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    df['PaceRank'] = df.groupby('RaceKey')['PredPace'].rank(method='min', ascending=True)
    df['ProbRank'] = df.groupby('RaceKey')['WinProb'].rank(method='min', ascending=False)
    
    # Common Filters
    df = df[df['Distance'] <= 600]
    df['MinOdds'] = df.groupby('RaceKey')['Odds'].transform('min')
    df = df[df['MinOdds'] >= 1.30] # No heavy fav races
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # Strategies
    # 1. Pace Leader (Baseline)
    strat_pace = df[(df['PaceRank'] == 1) & (df['CareerPrize'] >= 30000) & (df['Odds'] >= 1.50) & (df['Odds'] <= 30)].copy()
    
    # 2. Hybrid: Pace Leader AND High Probability (> 30%)
    # This filters out "Fast dogs" that the Prob Model hates (bad box/history)
    strat_hybrid = df[(df['PaceRank'] == 1) & (df['WinProb'] >= 0.30) & (df['CareerPrize'] >= 30000) & (df['Odds'] >= 1.50) & (df['Odds'] <= 30)].copy()
    
    # 3. Hybrid Value: Pace Leader AND Edge > 0 (Using Prob Model for Edge)
    df['Edge'] = (df['WinProb'] * df['Odds']) - 1
    strat_hybrid_value = df[(df['PaceRank'] == 1) & (df['Edge'] > 0.05) & (df['CareerPrize'] >= 30000) & (df['Odds'] >= 1.50) & (df['Odds'] <= 30)].copy()
    
    results = {
        'Pace Leader (Base)': strat_pace,
        'Hybrid (Pace + Prob>30%)': strat_hybrid,
        'Hybrid Value (Pace + Edge>5%)': strat_hybrid_value
    }
    
    print("\n" + "="*80)
    print("HYBRID STRATEGY RESULTS (2024-2025)")
    print("="*80)
    print(f"{'Strategy':<25} | {'Bets':<6} | {'Winners':<7} | {'Strike %':<8} | {'P/L':<8} | {'ROI %':<8}")
    print("-" * 80)
    
    for name, s_df in results.items():
        if len(s_df) == 0:
            print(f"{name:<25} | 0      | 0       | 0.0%     | 0.00     | 0.0%")
            continue
            
        bets = len(s_df)
        wins = s_df[s_df['Position'] == '1'].shape[0]
        strike = (wins / bets) * 100
        profit = s_df[s_df['Position'] == '1']['Odds'].sum() - bets
        roi = (profit / bets) * 100
        
        print(f"{name:<25} | {bets:<6} | {wins:<7} | {strike:<8.1f} | {profit:<8.1f} | {roi:<8.1f}")

if __name__ == "__main__":
    df = load_data_and_features()
    run_hybrid_backtest(df)
