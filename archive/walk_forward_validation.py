"""
Walk-Forward Validation for Greyhound ML Model

Tests model robustness by training on rolling 15-month windows
and testing on the following month.

This is the GOLD STANDARD for validating ML models and detecting overfitting.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from greyhound_ml_model import GreyhoundMLModel
import pickle

print("="*80)
print("WALK-FORWARD VALIDATION - PHASE 2")
print("="*80)
print("\nThis will train the model on rolling 15-month windows")
print("and test on out-of-sample months to validate robustness.\n")

# Define walk-forward schedule
# Each fold: 15 months training, 1 month testing
walk_forward_schedule = [
    # Training period (start, end), Test month
    (('2023-01-01', '2024-03-31'), '2024-04'),  # Train Jan 2023 - Mar 2024, Test Apr 2024
    (('2023-02-01', '2024-04-30'), '2024-05'),  # Train Feb 2023 - Apr 2024, Test May 2024
    (('2023-03-01', '2024-05-31'), '2024-06'),  # Train Mar 2023 - May 2024, Test Jun 2024
    (('2023-04-01', '2024-06-30'), '2024-07'),  # Train Apr 2023 - Jun 2024, Test Jul 2024
    (('2023-05-01', '2024-07-31'), '2024-08'),  # Train May 2023 - Jul 2024, Test Aug 2024
    (('2023-06-01', '2024-08-31'), '2024-09'),  # Train Jun 2023 - Aug 2024, Test Sep 2024
    (('2023-07-01', '2024-09-30'), '2024-10'),  # Train Jul 2023 - Sep 2024, Test Oct 2024
    (('2023-08-01', '2024-10-31'), '2024-11'),  # Train Aug 2023 - Oct 2024, Test Nov 2024
    (('2023-09-01', '2024-11-30'), '2024-12'),  # Train Sep 2023 - Nov 2024, Test Dec 2024
]

print(f"Walk-forward folds: {len(walk_forward_schedule)}")
print("\nSchedule:")
for i, ((train_start, train_end), test_month) in enumerate(walk_forward_schedule, 1):
    print(f"  Fold {i}: Train {train_start} to {train_end}, Test {test_month}")

# Store results for each fold
all_results = []

for fold_num, ((train_start, train_end), test_month) in enumerate(walk_forward_schedule, 1):
    print("\n" + "="*80)
    print(f"FOLD {fold_num}/{len(walk_forward_schedule)}: Testing {test_month}")
    print("="*80)

    # Calculate test month date range
    test_start = f"{test_month}-01"
    # Calculate last day of test month
    year, month = map(int, test_month.split('-'))
    if month == 12:
        next_month = f"{year+1}-01-01"
    else:
        next_month = f"{year}-{month+1:02d}-01"
    test_end = next_month

    print(f"\nTraining period: {train_start} to {train_end}")
    print(f"Testing period: {test_start} to {test_end}")

    # Train model on this fold's training data
    print("\n1. Extracting training data and training model...")
    model = GreyhoundMLModel()
    try:
        # Extract training data (returns DataFrame)
        train_df = model.extract_training_data(start_date=train_start, end_date=train_end)

        if train_df is None or len(train_df) == 0:
            print(f"   No training data for period {train_start} to {train_end}")
            continue

        # Prepare features
        X, y, train_df_with_features = model.prepare_features(train_df)

        if len(X) == 0:
            print(f"   No feature data after preparation")
            continue

        # Split for training (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model.train_model(X_train, y_train, X_test, y_test)

        print(f"   Model trained on {len(X_train):,} entries")

    except Exception as e:
        print(f"ERROR training model: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Get test data
    print(f"\n2. Loading test data for {test_month}...")
    conn = sqlite3.connect('greyhound_racing.db')

    query = """
    SELECT
        ge.EntryID,
        ge.GreyhoundID,
        g.GreyhoundName,
        t.TrackName,
        t.TrackID,
        rm.MeetingDate,
        r.RaceNumber,
        r.Distance,
        ge.Box,
        ge.Weight,
        ge.Position,
        ge.StartingPrice
    FROM GreyhoundEntries ge
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE rm.MeetingDate >= ? AND rm.MeetingDate < ?
    AND ge.Position IS NOT NULL
    AND ge.Position NOT IN ('DNF', 'SCR')
    ORDER BY rm.MeetingDate, t.TrackName, r.RaceNumber, ge.Box
    """

    df = pd.read_sql_query(query, conn, params=(test_start, test_end))
    print(f"   Loaded {len(df):,} test entries")

    # Filter NZ/TAS tracks
    excluded_tracks = ['Addington', 'Manukau', 'Hatrick', 'Cambridge', 'Palmerston North',
                      'Launceston', 'Hobart', 'Devonport']
    df = df[~df['TrackName'].isin(excluded_tracks)]
    df = df[~df['TrackName'].str.contains('NZ', na=False, case=False)]
    df = df[~df['TrackName'].str.contains('TAS', na=False, case=False)]
    print(f"   After filtering NZ/TAS: {len(df):,} entries")

    if len(df) == 0:
        print(f"   No test data for {test_month}, skipping...")
        conn.close()
        continue

    # Prepare test data
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['IsWinner'] = (df['Position'] == 1).astype(int)
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')

    print(f"   Winners: {df['IsWinner'].sum():,} ({df['IsWinner'].mean()*100:.2f}%)")

    # Load historical data for feature extraction
    print("\n3. Loading historical data for feature extraction...")
    historical_query = """
        SELECT
            ge.GreyhoundID,
            t.TrackName,
            r.Distance,
            ge.Weight,
            ge.Position,
            ge.FinishTime,
            ge.SplitBenchmarkLengths as G_Split_ADJ,
            rm.MeetingSplitAvgBenchmarkLengths as M_Split_ADJ,
            ge.FinishTimeBenchmarkLengths as G_OT_ADJ,
            rm.MeetingAvgBenchmarkLengths as M_OT_ADJ,
            rm.MeetingDate
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
          AND ge.SplitBenchmarkLengths IS NOT NULL
          AND rm.MeetingSplitAvgBenchmarkLengths IS NOT NULL
          AND ge.FinishTimeBenchmarkLengths IS NOT NULL
          AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
        ORDER BY ge.GreyhoundID, rm.MeetingDate DESC, r.RaceNumber DESC
    """

    hist_df = pd.read_sql_query(historical_query, conn, params=(test_start,))
    print(f"   Loaded {len(hist_df):,} historical races")

    # Calculate box win rates
    box_stats_query = """
        SELECT
            t.TrackID,
            r.Distance,
            ge.Box,
            COUNT(*) as TotalRaces,
            SUM(CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END) as Wins
        FROM GreyhoundEntries ge
        JOIN Races r ON ge.RaceID = r.RaceID
        JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
        JOIN Tracks t ON rm.TrackID = t.TrackID
        WHERE rm.MeetingDate < ?
          AND ge.Position IS NOT NULL
          AND ge.Box IS NOT NULL
        GROUP BY t.TrackID, r.Distance, ge.Box
    """
    box_stats_df = pd.read_sql_query(box_stats_query, conn, params=(test_start,))
    box_stats_df['BoxWinRate'] = box_stats_df['Wins'] / box_stats_df['TotalRaces']

    box_win_rates = {}
    for _, row in box_stats_df.iterrows():
        key = (row['TrackID'], row['Distance'], row['Box'])
        box_win_rates[key] = row['BoxWinRate']

    conn.close()

    # Extract features
    print("\n4. Extracting features...")
    hist_grouped = hist_df.groupby('GreyhoundID')

    features_list = []
    entry_info = []
    skipped = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"   Processed {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

        if row['GreyhoundID'] not in hist_grouped.groups:
            skipped += 1
            continue

        greyhound_hist = hist_grouped.get_group(row['GreyhoundID'])
        greyhound_hist = greyhound_hist[greyhound_hist['MeetingDate'] < row['MeetingDate']]
        last_5 = greyhound_hist.head(5)

        if len(last_5) < 5:
            skipped += 1
            continue

        features = {}

        # Box win rate
        box_key = (row['TrackID'], row['Distance'], row['Box'])
        features['BoxWinRate'] = box_win_rates.get(box_key, 0.125)

        # Recent form
        last_3 = last_5.head(3)
        if len(last_3) > 0:
            last_3_positions = pd.to_numeric(last_3['Position'], errors='coerce')
            features['AvgPositionLast3'] = last_3_positions.mean()
            features['WinRateLast3'] = (last_3_positions == 1).sum() / len(last_3)
        else:
            features['AvgPositionLast3'] = 4.5
            features['WinRateLast3'] = 0

        # GM_OT_ADJ features
        for i, (_, race) in enumerate(last_5.iterrows(), 1):
            gm_ot_adj = (race['G_OT_ADJ'] or 0) + (race['M_OT_ADJ'] or 0)
            track_weight = model.get_track_tier_weight(race['TrackName'])
            features[f'GM_OT_ADJ_{i}'] = gm_ot_adj * track_weight

        features_list.append(features)
        entry_info.append({
            'GreyhoundName': row['GreyhoundName'],
            'TrackName': row['TrackName'],
            'MeetingDate': row['MeetingDate'],
            'RaceNumber': row['RaceNumber'],
            'Box': row['Box'],
            'Won': row['IsWinner'],
            'Position': row['Position'],
            'StartingPrice': row['StartingPrice']
        })

    print(f"   Extracted features for {len(features_list):,} entries")
    print(f"   Skipped {skipped:,} entries")

    if len(features_list) == 0:
        print(f"   No features extracted for {test_month}, skipping...")
        continue

    # Make predictions
    print("\n5. Making predictions...")
    X = pd.DataFrame(features_list)
    entry_df = pd.DataFrame(entry_info)

    # Ensure all feature columns present
    missing_cols = set(model.feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[model.feature_columns]

    probabilities = model.model.predict_proba(X)[:, 1]
    entry_df['WinProbability'] = probabilities

    # Evaluate at 80% confidence threshold
    predictions_80 = entry_df[entry_df['WinProbability'] >= 0.8].copy()

    if len(predictions_80) > 0:
        winners = predictions_80['Won'].sum()
        win_rate = (winners / len(predictions_80)) * 100

        # Calculate ROI
        bets_with_odds = predictions_80[predictions_80['StartingPrice'].notna()].copy()
        if len(bets_with_odds) > 0:
            total_stake = len(bets_with_odds)
            winners_with_odds = bets_with_odds[bets_with_odds['Won'] == 1]
            total_return = winners_with_odds['StartingPrice'].sum() if len(winners_with_odds) > 0 else 0
            roi = ((total_return - total_stake) / total_stake) * 100
        else:
            roi = None
    else:
        win_rate = None
        roi = None

    # Store results
    result = {
        'fold': fold_num,
        'test_month': test_month,
        'train_start': train_start,
        'train_end': train_end,
        'total_test_entries': len(df),
        'predictions_80pct': len(predictions_80) if len(predictions_80) > 0 else 0,
        'winners_80pct': winners if len(predictions_80) > 0 else 0,
        'win_rate_80pct': win_rate,
        'roi_80pct': roi
    }
    all_results.append(result)

    print(f"\n6. Results for {test_month}:")
    print(f"   Predictions at 80%: {result['predictions_80pct']}")
    if win_rate is not None:
        print(f"   Win rate: {win_rate:.2f}%")
        print(f"   ROI: {roi:.2f}%" if roi is not None else "   ROI: N/A")
    else:
        print(f"   No predictions at 80% confidence")

# Summary report
print("\n" + "="*80)
print("WALK-FORWARD VALIDATION SUMMARY")
print("="*80)

results_df = pd.DataFrame(all_results)

print(f"\nFolds completed: {len(results_df)}")
print(f"\nResults by month (80% confidence threshold):\n")

for _, row in results_df.iterrows():
    if row['win_rate_80pct'] is not None:
        print(f"  {row['test_month']}: {row['predictions_80pct']:3d} bets, "
              f"{row['win_rate_80pct']:5.1f}% wins, "
              f"{row['roi_80pct']:+6.2f}% ROI")
    else:
        print(f"  {row['test_month']}: No predictions")

# Overall statistics
if len(results_df) > 0 and 'win_rate_80pct' in results_df.columns:
    valid_folds = results_df[results_df['win_rate_80pct'].notna()]
else:
    valid_folds = pd.DataFrame()

if len(valid_folds) > 0:
    print(f"\n" + "="*80)
    print("AGGREGATE STATISTICS (80% confidence)")
    print("="*80)

    total_predictions = valid_folds['predictions_80pct'].sum()
    total_winners = valid_folds['winners_80pct'].sum()
    avg_win_rate = (total_winners / total_predictions * 100) if total_predictions > 0 else 0
    avg_roi = valid_folds['roi_80pct'].mean()

    print(f"\nTotal predictions: {total_predictions:,}")
    print(f"Total winners: {total_winners:,}")
    print(f"Overall win rate: {avg_win_rate:.2f}%")
    print(f"Average ROI: {avg_roi:+.2f}%")
    print(f"ROI std dev: {valid_folds['roi_80pct'].std():.2f}%")
    print(f"Best month ROI: {valid_folds['roi_80pct'].max():+.2f}% ({valid_folds.loc[valid_folds['roi_80pct'].idxmax(), 'test_month']})")
    print(f"Worst month ROI: {valid_folds['roi_80pct'].min():+.2f}% ({valid_folds.loc[valid_folds['roi_80pct'].idxmin(), 'test_month']})")

    print(f"\n" + "="*80)
    print("VALIDATION ASSESSMENT")
    print("="*80)

    if avg_win_rate > 30 and avg_roi > 0:
        print("\n✅ EXCELLENT: Model shows consistent profitability across time periods")
        print("   The model is NOT overfit and generalizes well to new data.")
        print("   Proceed to live paper trading with confidence.")
    elif avg_win_rate > 25 and avg_roi > -5:
        print("\n⚠️  GOOD: Model shows good win rate but marginal profitability")
        print("   The model works but may need better bet selection criteria.")
        print("   Consider implementing edge-based filtering before live trading.")
    elif avg_win_rate > 20:
        print("\n⚠️  MODERATE: Model beats baseline but isn't profitable")
        print("   The model has predictive power but pricing is an issue.")
        print("   Focus on finding value bets where model >> market odds.")
    else:
        print("\n❌ POOR: Model doesn't consistently outperform baseline")
        print("   Possible overfitting or model needs fundamental improvements.")
        print("   Do NOT proceed to live trading without model redesign.")

else:
    print("\n⚠️  WARNING: No valid folds with predictions")
    print("   Model may be too conservative or data issues exist.")

print("\n" + "="*80)
print("Walk-forward validation complete!")
print("="*80)
