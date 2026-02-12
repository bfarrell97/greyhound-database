import pandas as pd
import numpy as np
import sqlite3
import joblib
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def load_and_filter_bets():
    print("Loading Data (2024-2025) for Strategy Validation...")
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= '2024-01-01'
    ORDER BY rm.MeetingDate ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    fe = FeatureEngineerV41()
    df = fe.calculate_features(df)
    features = fe.get_feature_list()
    
    df_clean = df.dropna(subset=features).copy()
    
    # Predict
    model = joblib.load('models/xgb_v41_final.pkl')
    dtest = xgb.DMatrix(df_clean[features])
    df_clean['Prob'] = model.predict(dtest)
    df_clean['ImpliedProb'] = 1.0 / df_clean['BSP']
    df_clean['Edge'] = df_clean['Prob'] - df_clean['ImpliedProb']
    
    # Golden Strategy Filters
    # Edge > 0.29, Prob > 0.35, $2.00 <= Price <= $7.00
    bets = df_clean[
        (df_clean['BSP'] >= 2.00) & 
        (df_clean['BSP'] <= 7.00) &
        (df_clean['Edge'] > 0.29) &
        (df_clean['Prob'] > 0.35)
    ].copy()
    
    bets['Win'] = (bets['win'] == 1).astype(int)
    stake = 10
    bets['Profit'] = np.where(bets['Win'] == 1, stake * (bets['BSP'] - 1) * 0.92, -stake)
    
    print(f"Total Bets Found: {len(bets)}")
    return bets

def run_k_fold(bets, k=10):
    print(f"\n--- {k}-Fold Cross Validation of Trades ---")
    bets = bets.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    fold_size = len(bets) // k
    
    rois = []
    
    print(f"{'Fold':<5} | {'Bets':<6} | {'Winners':<7} | {'Strike':<6} | {'Profit':<10} | {'ROI':<6}")
    print("-" * 60)
    
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(bets)
        fold = bets.iloc[start:end]
        
        profit = fold['Profit'].sum()
        turnover = len(fold) * 10
        roi = (profit / turnover) * 100
        strike = fold['Win'].mean() * 100
        
        rois.append(roi)
        print(f"{i+1:<5} | {len(fold):<6} | {fold['Win'].sum():<7} | {strike:>5.1f}% | ${profit:>9.2f} | {roi:+.1f}%")
        
    print("-" * 60)
    print(f"K-Fold Mean ROI: {np.mean(rois):.2f}%")
    print(f"K-Fold Std Dev:  {np.std(rois):.2f}%")
    return rois

def run_monte_carlo(bets, simulations=10000):
    print(f"\n--- Monte Carlo Simulation ({simulations} runs) ---")
    
    profits = bets['Profit'].values
    total_rois = []
    drawdowns = []
    
    for _ in range(simulations):
        # Bootstrap Resample
        resample = np.random.choice(profits, size=len(profits), replace=True)
        total_profit = np.sum(resample)
        total_turnover = len(resample) * 10
        roi = (total_profit / total_turnover) * 100
        total_rois.append(roi)
        
        # Max Drawdown for this sim
        cumulative = np.cumsum(resample)
        peak = np.maximum.accumulate(cumulative)
        dd = np.max(peak - cumulative)
        drawdowns.append(dd)
        
    # Stats
    mean_roi = np.mean(total_rois)
    percentile_5 = np.percentile(total_rois, 5)
    percentile_95 = np.percentile(total_rois, 95)
    prob_loss = np.mean(np.array(total_rois) < 0) * 100
    avg_dd = np.mean(drawdowns)
    max_dd_95 = np.percentile(drawdowns, 95)

    print(f"Mean ROI: {mean_roi:.2f}%")
    print(f"90% CI ROI: [{percentile_5:.2f}%, {percentile_95:.2f}%]")
    print(f"Probability of Valid Loss: {prob_loss:.2f}%") # Chance that actual ROI < 0
    print(f"Expected Max Drawdown: ${avg_dd:.2f}")
    print(f"Worst Case Drawdown (95%): ${max_dd_95:.2f}")

    if prob_loss < 5.0:
        print("\nPASS: Strategy is statistically significant (>95% confidence).")
    else:
        print("\nWARNING: Strategy has >5% chance of being luck.")

if __name__ == "__main__":
    bets = load_and_filter_bets()
    if len(bets) > 50:
        run_k_fold(bets, k=5)
        run_monte_carlo(bets)
    else:
        print("Not enough bets for rigorous validation.")
