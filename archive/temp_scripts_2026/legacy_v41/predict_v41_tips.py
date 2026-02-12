import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import sqlite3
import os
import sys
import re
from datetime import datetime, timedelta, timezone

sys.path.append(os.getcwd())
try:
    from src.features.feature_engineering_v41 import FeatureEngineerV41
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.features.feature_engineering_v41 import FeatureEngineerV41

def run_v41_predictions():
    """
    V41 SUPER MODEL: Handicapper Predictions
    Generates 'Value' tips based on Edge > 0.21
    """
    print("\n" + "="*80)
    print("V41 SUPER MODEL: HANDICAPPER PREDICTIONS")
    print("="*80)
    
    fe = FeatureEngineerV41()
    model = joblib.load('models/xgb_v41_final.pkl')
    features = fe.get_feature_list()
    
    from src.integration.betfair_fetcher import BetfairOddsFetcher
    fetcher = BetfairOddsFetcher()
    if not fetcher.login():
        return []

    markets = fetcher.get_greyhound_markets()
    if not markets:
        fetcher.logout()
        return []

    # Get today's historical candidate data for feature engineering
    conn = sqlite3.connect('greyhound_racing.db')
    query = """
    SELECT 
        ge.EntryID, ge.RaceID, ge.GreyhoundID, ge.Box, 
        ge.Position, ge.FinishTime, ge.Split, 
        ge.BSP, ge.Weight, ge.Margin, ge.TrainerID,
        r.Distance, r.Grade, t.TrackName, rm.MeetingDate,
        g.GreyhoundName, g.DateWhelped
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE rm.MeetingDate >= date('now', '-180 days')
    """
    # Note: Using 180 days of recent history to build rolling averages for today's runners
    df_history = pd.read_sql_query(query, conn)
    conn.close()
    
    tips = []
    skipped_form = 0
    skipped_price = 0
    processed_runners = 0
    
    print(f"Checking {len(markets)} markets...")
    for m in markets:
        try:
            m_name = m.market_name.upper()
            # Skip Place markets
            if any(x in m_name for x in ['PLACE', 'TO BE PLACED']):
                continue
                
            m_odds = fetcher.get_market_odds(m.market_id) or {}
            start_time = m.market_start_time.replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')
            print(f"  [DEBUG] Processing market: {m_name} | Start: {start_time} | Runners: {len(m.runners)}")
            
            # Map runners
            for r in m.runners:
                try:
                    processed_runners += 1
                    name = re.sub(r'^\d+\.\s*', '', r.runner_name).strip().upper()
                    price = m_odds.get(r.selection_id)
                    
                    # Filter out extremely short prices if they exist
                    if price and price <= 1.50:
                        skipped_price += 1
                        continue
                    
                    # Get dog-specific info from history
                    dog_hist = df_history[df_history['GreyhoundName'].str.upper() == name].copy()
                    if len(dog_hist) < 3: 
                        skipped_form += 1
                        continue
                    
                    dog_id = dog_hist.iloc[0]['GreyhoundID']
                    trainer_id = dog_hist.iloc[0]['TrainerID']
                    whelped = dog_hist.iloc[0]['DateWhelped']
                    
                    # Setup current race data for the runner
                    box_metadata = r.metadata.get('TRAP') if r.metadata else None
                    if box_metadata:
                        box = int(box_metadata)
                    else:
                        # Fallback: Parse from name "1. Name"
                        match = re.match(r'^(\d+)\.\s*', r.runner_name)
                        box = int(match.group(1)) if match else 0
                    
                    track = m.event.name.split(' ')[0] if hasattr(m, 'event') else "Unknown"
                    dist_match = re.search(r'(\d+)m', m.market_name)
                    dist = int(dist_match.group(1)) if dist_match else 400
                    
                    # Mock a current entry for prediction
                    current_entry = {
                        'EntryID': 999999, 'RaceID': 999999, 'GreyhoundID': dog_id, 'Box': box,
                        'Position': None, 'FinishTime': 0, 'Split': 0, 'BSP': price or 10.0,
                        'Weight': dog_hist.iloc[-1]['Weight'], 'Margin': 0, 'TrainerID': trainer_id, 
                        'Distance': dist, 'Grade': 'N/A', 'TrackName': track, 
                        'MeetingDate': datetime.now().strftime('%Y-%m-%d'),
                        'GreyhoundName': name, 'DateWhelped': whelped
                    }
                    
                    # Combine with history to calculate lags
                    full_df = pd.concat([dog_hist, pd.DataFrame([current_entry])]).reset_index(drop=True)
                    df_feat = fe.calculate_features(full_df)
                    
                    # Predict
                    X = df_feat.iloc[-1:][features]
                    dmatrix = xgb.DMatrix(X)
                    prob = model.predict(dmatrix)[0]
                    
                    rated_price = 1.0 / prob if prob > 0 else 99.0
                    
                    # If no live odds yet, use a high placeholder (10.0) for the Edge calculation 
                    current_price = price if price else 10.0
                    edge = prob - (1.0 / current_price)
                    
                    # DIAGNOSTIC: Print top runners even if not tips
                    if not hasattr(run_v41_predictions, 'top_probs'):
                        run_v41_predictions.top_probs = []
                    
                    feat_vals = {f: round(float(X[f].iloc[0]), 4) for f in ['RunTimeNorm_Lag3', 'Dog_Win_Rate', 'DogAgeDays']}
                    run_v41_predictions.top_probs.append((name, prob, price, edge, feat_vals))
                    
                    # "Action" Strategy: Edge > 0.21, Prob > 0.28, Price < $7.90
                    if edge > 0.21 and prob > 0.28 and (not price or price < 7.90):
                        tips.append({
                            'RaceTime': m.market_start_time.replace(tzinfo=timezone.utc).astimezone().strftime('%H:%M'),
                            'Box': box,
                            'Dog': name,
                            'Race': m.market_name,
                            'Track': track,
                            'Strategy': 'V41 Value',
                            'MarketPrice': price,
                            'RatedPrice': round(rated_price, 2),
                            'Stake': 10.0,
                            'Status': 'Ready',
                            'BetType': 'BACK',
                            'ModelProb': prob
                        })
                        print(f"🎯 [VALUE] {name} @ {track} - Prob: {prob:.2f} (Rated ${rated_price:.2f}) | Market: ${price:.2f} | Edge: {edge*100:+.0f}%")
                except Exception as runner_e:
                    # print(f"    [ERROR] Skipping runner {name if 'name' in locals() else 'Unknown'}: {runner_e}")
                    continue

        except Exception as e:
            print(f"  [ERROR] Market loop failed: {e}")
            pass

    fetcher.logout()
    print(f"\nFinal Summary:")
    print(f"  Processed Runners: {processed_runners}")
    print(f"  Skipped (Low Form): {skipped_form}")
    print(f"  Skipped (Low Price): {skipped_price}")
    print(f"  Generated Tips: {len(tips)}")
    
    # DIAGNOSTIC: Print top 5 candidates by Edge
    if hasattr(run_v41_predictions, 'top_probs') and run_v41_predictions.top_probs:
        print("\nTop 5 Candidates by Edge:")
        top_5 = sorted(run_v41_predictions.top_probs, key=lambda x: x[3], reverse=True)[:5]
        for name, prob, price, edge, feats in top_5:
            price_str = f"${price:.2f}" if price else "N/A"
            print(f"  {name:20} | Prob: {prob:.4f} | Price: {price_str} | Edge: {edge*100:+.1f}% | {feats}")

    return tips

if __name__ == "__main__":
    t = run_v41_predictions()
    if t:
        print(pd.DataFrame(t))
