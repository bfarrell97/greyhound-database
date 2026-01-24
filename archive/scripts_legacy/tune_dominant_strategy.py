"""
Tune Dominant Leader Strategy
Granular grid search to optimize ROI/Volume for the "Predicted Gap" strategy.
Testing looser prize filters and varying odds/gap thresholds.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle

DB_PATH = 'greyhound_racing.db'
PACE_MODEL_PATH = 'models/pace_xgb_model.pkl'

def load_and_prep():
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
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    # Features
    print("Feature Eng...")
    pace_bench = df[df['FinishTime'] > 0].groupby(['TrackName', 'Distance'])['FinishTime'].median().reset_index()
    pace_bench.columns = ['TrackName', 'Distance', 'TrackDistMedianPace']
    df = df.merge(pace_bench, on=['TrackName', 'Distance'], how='left')
    df['NormTime'] = df['FinishTime'] - df['TrackDistMedianPace']
    
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    df['p_Roll3'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
    df['p_Roll5'] = g['NormTime'].transform(lambda x: x.shift(1).rolling(5, min_periods=5).mean())
    df['p_Lag1'] = g['NormTime'].shift(1)
    df['p_Lag2'] = g['NormTime'].shift(2)
    df['p_Lag3'] = g['NormTime'].shift(3)
    df['PrevDate'] = g['MeetingDate'].shift(1)
    df['DaysSince'] = (df['MeetingDate'] - df['PrevDate']).dt.days.fillna(999).clip(upper=60)
    df['CareerPrize'] = g['PrizeMoney'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    # Filter Valid
    df = df.dropna(subset=['p_Roll5', 'Odds']).copy()
    
    # Predict
    print("Predicting...")
    with open(PACE_MODEL_PATH, 'rb') as f: pace_model = pickle.load(f)
    cols = ['p_Lag1', 'p_Lag2', 'p_Lag3', 'p_Roll3', 'p_Roll5', 'DaysSince', 'Box', 'Distance']
    X = df[cols].copy()
    X.columns = ['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll5', 'DaysSince', 'Box', 'Distance']
    df['PredNormPace'] = pace_model.predict(X)
    df['PredPace'] = df['PredNormPace'] + df['TrackDistMedianPace']
    
    # Calculate Gaps
    print("Calculating Gaps...")
    df['RaceKey'] = df['TrackName'] + '_' + df['MeetingDate'].astype(str) + '_' + df['RaceID'].astype(str)
    
    # Only sufficiently large fields
    df['FieldSize'] = df.groupby('RaceKey')['Odds'].transform('count')
    df = df[df['FieldSize'] >= 6]
    
    # Sort for Gap Calc
    df = df.sort_values(['RaceKey', 'PredPace'])
    
    # Find Leader and 2nd
    # Use grouped apply or transform for speed? 
    # Transform is faster.
    # Prediction Rank 1 = Leader
    df['Rank'] = df.groupby('RaceKey').cumcount() + 1
    
    # We need the 2nd best time in the race attached to the leader
    # Since it's sorted by PredPace, the 2nd row in each group is the 2nd best time.
    # We can shift(-1) within group? No, shift(-1) on the Leader row gives the 2nd row's value.
    df['NextTime'] = df.groupby('RaceKey')['PredPace'].shift(-1)
    
    # Gap = NextTime - MyTime (Positive means I am faster)
    # Only valid for Rank 1
    df['Gap'] = df['NextTime'] - df['PredPace']
    
    return df

def tune(df):
    print("Running Grid Search...")
    leaders = df[df['Rank'] == 1].copy()
    
    prizes = [0, 5000, 10000, 20000]
    gaps = [0.10, 0.15, 0.20, 0.25, 0.30]
    min_odds = [1.50, 1.80, 2.00, 2.20, 2.50]
    dists = ['All', 'Short', 'Medium'] # Short < 400, Med >= 400
    
    results = []
    count = 0 
    total = len(prizes) * len(gaps) * len(min_odds) * len(dists)
    
    for prize in prizes:
        # Prize Filter
        cand_prize = leaders[leaders['CareerPrize'] >= prize]
        
        for d_cat in dists:
            # Dist Filter
            if d_cat == 'Short':
                cand_dist = cand_prize[cand_prize['Distance'] < 400]
            elif d_cat == 'Medium':
                cand_dist = cand_prize[(cand_prize['Distance'] >= 400) & (cand_prize['Distance'] <= 600)]
            else:
                cand_dist = cand_prize[cand_prize['Distance'] <= 600]
            
            for gap in gaps:
                # Gap Filter
                cand_gap = cand_dist[cand_dist['Gap'] >= gap]
                
                for odd in min_odds:
                    # Odds Filter
                    final = cand_gap[(cand_gap['Odds'] >= odd) & (cand_gap['Odds'] <= 30)]
                    
                    if len(final) < 50: continue
                    
                    bets = len(final)
                    wins = final[final['Position'] == '1'].shape[0]
                    strike = (wins / bets) * 100
                    profit = final[final['Position'] == '1']['Odds'].sum() - bets
                    roi = (profit / bets) * 100
                    
                    results.append({
                        'Prize': prize,
                        'Dist': d_cat,
                        'Gap': gap,
                        'MinOdds': odd,
                        'Bets': bets,
                        'Strike': strike,
                        'ROI': roi,
                        'Profit': profit
                    })
                    count += 1
                    if count % 50 == 0:
                        print(f"Tested {count}/{total}...", end='\r')
                        
    print("\nSearch Complete.")
    res_df = pd.DataFrame(results)
    
    # Filter for viable strategies (e.g. ROI > 0)
    profitable = res_df[res_df['ROI'] > 0].sort_values('Profit', ascending=False)
    
    print("\n" + "="*100)
    print("TOP PROFITABLE CONFIGURATIONS (Sorted by Total Profit)")
    print("="*100)
    if len(profitable) > 0:
        print(profitable.head(20).to_string(index=False))
    else:
        print("No profitable configurations found > 50 bets.")
        # Show best negative
        print("\nBest Negative ROI:")
        print(res_df.sort_values('ROI', ascending=False).head(10).to_string(index=False))
        
    res_df.to_csv('results/tuning_results.csv', index=False)
    print("\nSaved to results/tuning_results.csv")

if __name__ == "__main__":
    df = load_and_prep()
    tune(df)
