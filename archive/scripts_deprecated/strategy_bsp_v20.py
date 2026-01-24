"""
V20 Strategy Simulation
Use V20 Model (BSP Prediction) to find value bets.
Logic:
- Predict 'Fair BSP' using V20
- Compare with 'Price5Min' (Early Market)
- Back if Price5Min > PredictedBSP * (1 + Edge)
- Lay if Price5Min < PredictedBSP * (1 - Edge)
"""
import pandas as pd
import numpy as np
import sqlite3
from autogluon.tabular import TabularPredictor

# Load Model
MODEL_PATH = "models/autogluon_bsp_v20_1909"
predictor = TabularPredictor.load(MODEL_PATH)

# Feature Columns expected by model (Excluding Target)
# We must ensure we generate exactly the same features as V20 training
FEATURES = predictor.feature_metadata_in.get_features()

print(f"Loaded Model: {MODEL_PATH}")
print(f"Expected Features: {len(FEATURES)}")

# Load Test Data (2024-2025 where Price5Min exists)
conn = sqlite3.connect('greyhound_racing.db')
query = """
SELECT ge.RaceID, ge.GreyhoundID, ge.Position, ge.BSP, ge.Price5Min, ge.Box,
       ge.FinishTime, ge.Split, ge.Weight,
       ge.TrainerID, r.Distance, rm.MeetingDate, t.TrackName, rm.TrackID,
       g.SireID
FROM GreyhoundEntries ge
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
LEFT JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
WHERE rm.MeetingDate >= '2024-01-01'
  AND ge.Price5Min IS NOT NULL
  AND ge.BSP > 1
ORDER BY rm.MeetingDate
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
df['Year'] = df['MeetingDate'].dt.year
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['LogBSP'] = np.log(df['BSP'].clip(1.01, 500))
df['HasPrice5Min'] = 1
df['LogPrice5Min'] = np.log(df['Price5Min'])

print(f"Loaded {len(df):,} test candidates")

# --- FEATURE GENERATION (Must Match V20) ---
from collections import defaultdict
dog_hist = defaultdict(list)
trainer_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
sire_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})
box_stat = defaultdict(lambda: {'runs': 0, 'bsp_sum': 0})

rows = []
for race_id, race_df in df.groupby('RaceID', sort=False):
    for _, r in race_df.iterrows():
        dog_id = r['GreyhoundID']
        hist = dog_hist.get(dog_id, [])
        
        # Always generate row if we have history
        if len(hist) >= 3:
            feat = r.to_dict()
            feat['LogPrice5Min'] = np.log(r['Price5Min'])
            
            # Replicate V20 Logic
            bsps = [h['bsp'] for h in hist[-10:]]
            if len(bsps) >= 3:
                feat['BSP_Mean_10'] = np.mean(bsps)
                feat['BSP_Max_10'] = max(bsps)
                feat['BSP_Min_10'] = min(bsps)
                feat['BSP_Std_10'] = np.std(bsps)
                
                bsps5 = bsps[-5:]
                feat['BSP_Mean_5'] = np.mean(bsps5)
                feat['BSP_Max_5'] = max(bsps5)
                feat['BSP_Min_5'] = min(bsps5)
                feat['BSP_Std_5'] = np.std(bsps5)
                
                bsps3 = bsps[-3:]
                feat['BSP_Mean_3'] = np.mean(bsps3)
                feat['BSP_Max_3'] = max(bsps3)
                feat['BSP_Min_3'] = min(bsps3)
                feat['BSP_Std_3'] = np.std(bsps3)
                
                # Pos Stats
                pos = [h['pos'] for h in hist[-10:] if h['pos']]
                if len(pos) >= 3:
                    feat['Pos_Mean_10'] = np.mean(pos)
                    feat['Pos_Mean_5'] = np.mean(pos[-5:])
                    feat['Pos_Mean_3'] = np.mean(pos[-3:])
                    feat['WinRate_10'] = sum(1 for p in pos if p==1)/len(pos)
                else:
                    feat['Pos_Mean_10'] = 4.5
                    feat['Pos_Mean_5'] = 4.5
                    feat['Pos_Mean_3'] = 4.5
                    feat['WinRate_10'] = 0

            # Context
            tid = r['TrainerID']
            feat['Trainer_AvgBSP'] = trainer_stat[tid]['bsp_sum']/trainer_stat[tid]['runs'] if trainer_stat[tid]['runs'] > 5 else 10
            
            sid = r['SireID']
            feat['Sire_AvgBSP'] = sire_stat[sid]['bsp_sum']/sire_stat[sid]['runs'] if sid and sire_stat[sid]['runs'] > 10 else 10
            
            bid = (r['TrackID'], r['Box'])
            feat['Box_Track_AvgBSP'] = box_stat[bid]['bsp_sum']/box_stat[bid]['runs'] if box_stat[bid]['runs'] > 5 else 10
            
            feat['BSP_Trend_5'] = hist[-1]['bsp'] - hist[-5]['bsp'] if len(hist) >= 5 else 0

            rows.append(feat)

    # Update History
    for _, r in race_df.iterrows():
        dog_hist[r['GreyhoundID']].append({'bsp': r['BSP'] if r['BSP'] else 10, 'pos': pd.to_numeric(r['Position'], errors='coerce')})
        trainer_stat[r['TrainerID']]['runs'] += 1; trainer_stat[r['TrainerID']]['bsp_sum'] += (r['BSP'] if r['BSP'] else 10)
        if r['SireID']: sire_stat[r['SireID']]['runs'] += 1; sire_stat[r['SireID']]['bsp_sum'] += (r['BSP'] if r['BSP'] else 10)
        box_stat[(r['TrackID'], r['Box'])]['runs'] += 1; box_stat[(r['TrackID'], r['Box'])]['bsp_sum'] += (r['BSP'] if r['BSP'] else 10)

test_df = pd.DataFrame(rows)
print(f"Generated Features for {len(test_df):,} rows")

# Prediction
print("Predicting...")
test_df['PredLogBSP'] = predictor.predict(test_df[FEATURES])
test_df['PredBSP'] = np.exp(test_df['PredLogBSP'])

# Strategy Simulation
edge = 0.20 # 20% Edge required
print(f"\nSIMULATION (Edge {edge*100}%)")

# BACK: Price5Min > Fair Price (Market Underestimates)
backs = test_df[test_df['Price5Min'] > test_df['PredBSP'] * (1 + edge)].copy()
backs_wins = len(backs[backs['Position'] == 1])
backs_profit = backs[backs['Position'] == 1]['Price5Min'].sum() * 0.95 - len(backs) # Commission 5%? User said 10%
# User said 10% comm on net winnings? Or standard Betfair 5%?
# User prompt: "minimum 5% ROI with 10% commission"
COMM = 0.10
backs_ret = backs[backs['Position'] == 1]['Price5Min'].sum()
backs_profit = (backs_ret - len(backs)) * (1 - COMM) # Profit after comm

print(f"\nBACK STRATEGY:")
print(f"Bets: {len(backs):,}")
print(f"Wins: {backs_wins:,} ({backs_wins/len(backs)*100:.1f}%)")
print(f"Profit: ${backs_profit:.2f}")
print(f"ROI: {backs_profit/len(backs)*100:.2f}%")

# LAY: Price5Min < Fair Price (Market Overestimates)
lays = test_df[test_df['Price5Min'] < test_df['PredBSP'] * (1 - edge)].copy()
# Lay Liability is strictly controlled? Or Fixed Stake? 
# Fixed Liability Laying is safest.
# Profit = Stake * 0.95 (if wins) - (Liability) (if loses)
# Wait, standard lay:
# If dog LOSES (pos!=1): We win Stake * (1-Comm)
# If dog WINS (pos==1): We lose Stake * (Price - 1)
lays_wins = len(lays[lays['Position'] != 1]) # We win the bet if dog loses
lays_losses = len(lays[lays['Position'] == 1])
liability_lost = lays[lays['Position'] == 1].apply(lambda x: x['Price5Min'] - 1, axis=1).sum()
gross_profit = lays_wins * 1 # $1 stake per lay
net_profit_lays = (gross_profit - liability_lost) 
# Commission on NET profit per market? Or per bet? Usually per market.
# Simple approx: deduct comm from positive profit.

print(f"\nLAY STRATEGY:")
print(f"Bets: {len(lays):,}")
print(f"Successful Lays: {lays_wins:,} ({lays_wins/len(lays)*100:.1f}%)")
print(f"Liability Lost: ${liability_lost:.2f}")
print(f"Net Profit (Est): ${net_profit_lays:.2f}")
print(f"ROI: {net_profit_lays/len(lays)*100:.2f}%")

# Accuracy Check
mape = np.mean(np.abs(test_df['BSP'] - test_df['PredBSP']) / test_df['BSP']) * 100
print(f"\nModel MAPE: {mape:.1f}%")

