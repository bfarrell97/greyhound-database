"""
ML Strategy Optimizer (Grid Search)
Iterates through combinations of filters to find profitable ML strategies.
"""

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import itertools

DB_PATH = 'greyhound_racing.db'
PIR_MODEL_PATH = 'models/pir_xgb_model.pkl'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'
PROB_MODEL_PATH = 'models/prob_xgb_model.pkl'

def load_data_and_predict():
    conn = sqlite3.connect(DB_PATH)
    print("Loading Data (2024-2025)...")
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
    
    # Feature Eng
    print("Feature Engineering...")
    split_bench = df[df['Split'] > 0].groupby(['TrackName', 'Distance'])['Split'].median().reset_index()
    split_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianSplit']
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    
    df = df.merge(split_bench, on=['TrackName', 'Distance'], how='left')
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    
    df['NormSplit'] = df['Split'] - df['TrackDistMedianSplit']
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    
    # Rolling Features
    g = df.groupby('GreyhoundID')
    
    # SPLIT
    df['s_Roll3'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['s_Roll5'] = g['NormSplit'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['s_Lag1'] = g['NormSplit'].shift(1)
    df['s_Lag2'] = g['NormSplit'].shift(2)
    df['s_Lag3'] = g['NormSplit'].shift(3)
    
    # PACE
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    
    # Career Prize
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter valid history
    df = df.dropna(subset=['s_Roll5', 'p_Roll5', 'Odds']).copy()
    
    # Predict
    print("Loading Models & Predicting...")
    with open(PIR_MODEL_PATH, 'rb') as f: pir_model = pickle.load(f)
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    with open(PROB_MODEL_PATH, 'rb') as f: prob_model = pickle.load(f)
    
    # Prepare X
    cols_split = ['s_Lag1', 's_Lag2', 's_Lag3', 's_Roll3', 's_Roll5', 'DaysSince', 'Box', 'Distance']
    cols_pace = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    cols_prob = ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    
    X_split = df[cols_split].copy()
    X_split.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance'] # Rename for model
    df['PredSplit'] = pir_model.predict(X_split)
    
    X_pace = df[cols_pace].copy()
    X_pace.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredPace'] = pace_model.predict(X_pace)
    
    # Prob Model expects specific col names? Let's check training script.
    # Training feat names: ['s_Roll3', 's_Roll5', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    # If the pickle saved feature names, we must match.
    # The dataframe X_prob already has these names.
    X_prob = df[cols_prob].copy()
    df['WinProb'] = prob_model.predict_proba(X_prob)[:, 1]
    
    # Denormalize (Rank needs absolute or relative, but for grouping per race, relative works)
    # But strictly, ranks should be per race.
    # Let's add race keys and rank.
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Create Ranks
    df['PaceRank'] = df.groupby('RaceKey')['PredPace'].rank(method='min', ascending=True)
    df['SplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min', ascending=True)
    
    return df

def optimize(df):
    print("Running Grid Search...")
    
    # Parameters
    prizes = [0, 5000, 10000, 20000, 30000]
    min_odds_list = [1.20, 1.50, 2.00]
    prob_thresholds = [0.0, 0.20, 0.30, 0.40]
    bases = ['Pace Leader', 'Dual Leader', 'Split Leader']
    
    results = []
    
    # Base Filters (Common)
    df_base = df[df['Distance'] <= 600].copy()
    df_base['FieldSize'] = df_base.groupby('RaceKey')['Odds'].transform('count')
    df_base = df_base[df_base['FieldSize'] >= 6]
    df_base['MinRaceOdds'] = df_base.groupby('RaceKey')['Odds'].transform('min')
    
    # For speed, pre-calculate masks
    mask_pace = df_base['PaceRank'] == 1
    mask_split = df_base['SplitRank'] == 1
    
    total_iterations = len(bases) * len(prizes) * len(min_odds_list) * len(prob_thresholds)
    count = 0
    
    for base in bases:
        # Select Candidate Rows based on Strategy
        if base == 'Pace Leader':
            candidates = df_base[mask_pace]
        elif base == 'Split Leader':
            candidates = df_base[mask_split]
        elif base == 'Dual Leader':
            candidates = df_base[mask_pace & mask_split]
            
        for prize in prizes:
            # Apply Prize Filter
            cand_prize = candidates[candidates['CareerPrize'] >= prize]
            
            for min_odd in min_odds_list:
                # Apply Odds Filter
                cand_odds = cand_prize[(cand_prize['Odds'] >= min_odd) & (cand_prize['Odds'] <= 30)]
                # Also check race wasn't too heavy fav
                cand_odds = cand_odds[cand_odds['MinRaceOdds'] >= 1.30]
                
                for prob in prob_thresholds:
                    # Apply Prob Filter
                    final_set = cand_odds[cand_odds['WinProb'] >= prob]
                    
                    if len(final_set) < 100: continue
                    
                    # Calc Metrics
                    bets = len(final_set)
                    wins = final_set[final_set['Position'] == '1'].shape[0]
                    strike = (wins / bets) * 100
                    profit = final_set[final_set['Position'] == '1']['Odds'].sum() - bets
                    roi = (profit / bets) * 100
                    
                    results.append({
                        'Strategy': base,
                        'Prize': prize,
                        'MinOdds': min_odd,
                        'MinProb': prob,
                        'Bets': bets,
                        'Strike': strike,
                        'ROI': roi,
                        'Profit': profit
                    })
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"Processed {count}/{total_iterations} combinations...", end='\r')
                        
    print("\nSearch Complete.")
    
    # Convert to DF and Sort
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('ROI', ascending=False)
    
    print("\n" + "="*100)
    print("TOP 20 PROFITABLE CONFIGURATIONS (2024-2025)")
    print("="*100)
    print(res_df.head(20).to_string(index=False))
    
    # Save detailed CSV
    res_df.to_csv('results/optimization_results.csv', index=False)
    print("\nFull results saved to results/optimization_results.csv")

if __name__ == "__main__":
    df = load_data_and_predict()
    optimize(df)
