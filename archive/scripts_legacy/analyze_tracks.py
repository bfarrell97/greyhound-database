"""
Analyze Tracks for Profitable Pockets (2025)
Goal: Identify tracks where Race Shape strategies (Dominator, Swooper) are profitable.
"""
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

DB_PATH = 'greyhound_racing.db'

def load_data():
    print("Loading Data (2023-2025)...")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ge.GreyhoundID,
        g.SireID,
        r.RaceID,
        rm.MeetingDate,
        t.TrackName,
        r.Distance,
        ge.Box,
        ge.FinishTime,
        ge.Split,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= '2023-01-01'
      AND ge.Position NOT IN ('DNF', 'SCR', '')
      AND ge.FinishTime IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
    df['Box'] = pd.to_numeric(df['Box'], errors='coerce').fillna(0)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0)
    df['Split'] = pd.to_numeric(df['Split'], errors='coerce')
    df['IsWin'] = (df['Position'] == '1').astype(int)
    
    # Run Home
    df['RunHome'] = df['FinishTime'] - df['Split']
    
    def parse_price(x):
        try:
            if not x: return np.nan
            x = str(x).replace('$', '').strip().replace('F', '')
            return float(x)
        except: return np.nan
    df['Odds'] = df['StartingPrice'].apply(parse_price)
    
    return df

def feature_engineering(df):
    print("Engineering Features...")
    
    # 1. Benchmarks (Global)
    cols = ['FinishTime', 'Split', 'RunHome']
    for c in cols:
        bench_col = f'TrackDistMedian{c}'
        bench = df.groupby(['TrackName', 'Distance'])[c].median().reset_index()
        bench.columns = ['TrackName', 'Distance', bench_col]
        df = df.merge(bench, on=['TrackName', 'Distance'], how='left')
        df[f'Norm{c}'] = df[c] - df[bench_col]
        
    # 2. Rolling Avgs (Dog)
    df = df.sort_values(['GreyhoundID', 'MeetingDate'])
    g = df.groupby('GreyhoundID')
    
    for c in ['NormFinishTime', 'NormSplit', 'NormRunHome']:
        df[f'Dog{c}Avg'] = g[c].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
    # 3. Sire Stats
    df = df.sort_values(['SireID', 'MeetingDate'])
    df['SireWinRate'] = df.groupby('SireID')['IsWin'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0)
    
    df = df.dropna(subset=['Split', 'FinishTime', 'RunHome']).copy()
    return df

def train_and_predict(df):
    train_mask = (df['MeetingDate'].dt.year >= 2023) & (df['MeetingDate'].dt.year <= 2024)
    test_mask = (df['MeetingDate'].dt.year == 2025)
    
    train_df = df[train_mask]
    test_df = df[test_mask].copy()
    
    print(f"Training Size: {len(train_df)}")
    
    features = ['DogNormFinishTimeAvg', 'DogNormSplitAvg', 'DogNormRunHomeAvg', 'SireWinRate', 'Box', 'Distance']
    targets = {'Split': 'NormSplit', 'RunHome': 'NormRunHome', 'Overall': 'NormFinishTime'}
    
    for name, target in targets.items():
        print(f"  Training {name} Model...")
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=100, learning_rate=0.1, max_depth=3, n_jobs=-1
        )
        t_df = train_df.dropna(subset=[target])
        model.fit(t_df[features], t_df[target])
        test_df[f'Pred{name}'] = model.predict(test_df[features])

    return test_df

def analyze_tracks(df):
    print("\n" + "="*80)
    print("TRACK SPECIFIC ANALYSIS (2025)")
    print("="*80)
    
    df['RaceKey'] = df['MeetingDate'].astype(str) + '_' + df['TrackName'] + '_' + df['RaceID'].astype(str)
    
    # Ranks
    df['PredSplitRank'] = df.groupby('RaceKey')['PredSplit'].rank(method='min')
    df['PredRunHomeRank'] = df.groupby('RaceKey')['PredRunHome'].rank(method='min')
    df['PredOverallRank'] = df.groupby('RaceKey')['PredOverall'].rank(method='min')
    
    # Define Strategies
    df['IsDominator'] = (df['PredSplitRank'] == 1) & (df['PredOverallRank'] == 1)
    df['IsSwooper'] = (df['PredRunHomeRank'] == 1) & (df['PredOverallRank'] == 1) & (df['PredSplitRank'] > 1)
    df['IsRabbit'] = (df['PredSplitRank'] == 1) # Pure Early Speed
    df['IsSteamroller'] = (df['PredRunHomeRank'] == 1) # Pure Late Speed
    
    # Group by Track
    tracks = df['TrackName'].unique()
    results = []
    
    print(f"{'Track':<25} {'Strat':<12} {'Bets':<6} {'Wins':<6} {'Strike%':<8} {'Profit':<8} {'ROI%':<8}")
    print("-" * 80)
    
    for track in tracks:
        track_df = df[df['TrackName'] == track]
        
        strategies = {
            'Dominator': track_df['IsDominator'],
            'Swooper': track_df['IsSwooper'],
            'Rabbit': track_df['IsRabbit'],
            'Steamroller': track_df['IsSteamroller']
        }
        
        for strat_name, mask in strategies.items():
            bets = track_df[mask & (track_df['Odds'] >= 2.0) & (track_df['Odds'] <= 30)].copy()
            
            if len(bets) >= 20:
                wins = bets['IsWin'].sum()
                profit = (bets[bets['IsWin'] == 1]['Odds'] - 1).sum() - (len(bets) - wins)
                roi = profit / len(bets) * 100
                
                if roi > -5:
                     print(f"{track:<25} {strat_name:<12} {len(bets):<6} {wins:<6} {wins/len(bets)*100:<8.1f} {profit:<8.1f} {roi:<8.1f}")
                
                results.append({
                    'Track': track, 
                    'Strategy': strat_name, 
                    'Bets': len(bets), 
                    'Wins': wins, 
                    'Strike': wins/len(bets), 
                    'Profit': profit, 
                    'ROI': roi
                })

    # Save CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROI', ascending=False)
    results_df.to_csv('track_analysis.csv', index=False)
    print("\nFull results saved to track_analysis.csv")

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    test_df = train_and_predict(df)
    analyze_tracks(test_df)
