"""
Find dogs the market is mispricing - where our model says high confidence
but market odds suggest lower win probability.

Uses 70% Pace + 30% Form @ 0.80+ confidence on $1.50-$3.00 odds
"""

import sqlite3
import pandas as pd
from datetime import datetime

def progress(msg, indent=0):
    ts = datetime.now().strftime('%H:%M:%S')
    prefix = "  " * indent
    print(f"[{ts}] {prefix}{msg}", flush=True)

progress("="*120)
progress("FINDING MISPRICE OPPORTUNITIES: Dogs market underestimates vs our model")
progress("="*120)

DB_PATH = 'greyhound_racing.db'
conn = sqlite3.connect(DB_PATH)

# Get all unique race dates in last 3 months
query_dates = """
SELECT DISTINCT DATE(rm.MeetingDate) as RaceDate
FROM Races r
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
WHERE rm.MeetingDate >= '2025-10-01' AND rm.MeetingDate <= '2025-12-31'
ORDER BY RaceDate
"""

progress("\nGetting all race dates in last 3 months (Oct-Dec)...", indent=1)
dates_df = pd.read_sql_query(query_dates, conn)
race_dates = dates_df['RaceDate'].tolist()
progress(f"Found {len(race_dates)} unique race dates", indent=1)

all_bets = []
total_processed = 0

progress("\nSimulating daily predictions and tracking results...", indent=1)
progress(f"Processing {len(race_dates)} dates (this will take several minutes)...\n", indent=1)

for idx, race_date in enumerate(race_dates, 1):
    if idx % 20 == 0:
        progress(f"[{idx}/{len(race_dates)}] Processing {race_date}... ({total_processed} runners so far)", indent=1)

    # Get all races for this date
    query = """
    SELECT r.RaceID, r.Distance,
           ge.GreyhoundID, g.GreyhoundName, ge.StartingPrice, ge.FinishTimeBenchmarkLengths,
           rm.MeetingAvgBenchmarkLengths, ge.Position
    FROM Races r
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN GreyhoundEntries ge ON r.RaceID = ge.RaceID
    JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
    WHERE DATE(rm.MeetingDate) = ?
      AND ge.Position NOT IN ('DNF', 'SCR')
      AND ge.StartingPrice BETWEEN 1.5 AND 3.0
    """

    df = pd.read_sql_query(query, conn, params=[race_date])

    if len(df) == 0:
        continue

    # Convert prices to numeric
    df['StartingPrice'] = pd.to_numeric(df['StartingPrice'], errors='coerce')
    df = df.dropna(subset=['StartingPrice'])

    # Group by race to calculate scores within each race
    for race_id, race_group in df.groupby('RaceID'):
        # Calculate pace metrics BEFORE this date for each dog in this race
        pace_query = """
        WITH dog_pace_history AS (
            SELECT ge.GreyhoundID,
                   (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench,
                   ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE rm.MeetingDate < ? AND ge.Position NOT IN ('DNF', 'SCR')
        ),
        dog_pace_avg AS (
            SELECT GreyhoundID,
                   AVG(CASE WHEN RaceNum <= 5 THEN TotalFinishBench END) as HistoricalPaceAvg
            FROM dog_pace_history
            GROUP BY GreyhoundID
            HAVING COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) >= 5
        )
        SELECT ge.GreyhoundID, COALESCE(dpa.HistoricalPaceAvg, 0) as HistoricalPaceAvg
        FROM (SELECT DISTINCT GreyhoundID FROM GreyhoundEntries WHERE RaceID = ?) ge
        LEFT JOIN dog_pace_avg dpa ON ge.GreyhoundID = dpa.GreyhoundID
        """

        pace_df = pd.read_sql_query(pace_query, conn, params=[race_date, race_id])

        # Calculate form metrics BEFORE this date for each dog
        form_query = """
        WITH dog_form_data AS (
            SELECT ge.GreyhoundID, 
                   CASE WHEN ge.Position = '1' THEN 1 ELSE 0 END as IsWin,
                   ROW_NUMBER() OVER (PARTITION BY ge.GreyhoundID ORDER BY rm.MeetingDate DESC) as RaceNum
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            WHERE rm.MeetingDate < ? AND ge.Position NOT IN ('DNF', 'SCR')
        )
        SELECT GreyhoundID,
               SUM(CASE WHEN RaceNum <= 5 THEN IsWin ELSE 0 END) as FormWins,
               COUNT(CASE WHEN RaceNum <= 5 THEN 1 END) as FormRaces
        FROM dog_form_data
        GROUP BY GreyhoundID
        """

        form_df = pd.read_sql_query(form_query, conn, params=[race_date])
        form_df['RecentWinRate'] = (form_df['FormWins'] / form_df['FormRaces'].clip(lower=1)) * 100

        # Merge metrics
        race_metrics = race_group.merge(pace_df, on='GreyhoundID', how='left')
        race_metrics = race_metrics.merge(form_df[['GreyhoundID', 'RecentWinRate']], on='GreyhoundID', how='left')
        race_metrics['RecentWinRate'] = race_metrics['RecentWinRate'].fillna(0)
        race_metrics['HistoricalPaceAvg'] = race_metrics['HistoricalPaceAvg'].fillna(0)

        # Normalize scores within this race
        pace_min = race_metrics['HistoricalPaceAvg'].min()
        pace_max = race_metrics['HistoricalPaceAvg'].max()

        if pace_max - pace_min == 0:
            race_metrics['PaceScore'] = 0.5
        else:
            race_metrics['PaceScore'] = (race_metrics['HistoricalPaceAvg'] - pace_min) / (pace_max - pace_min)

        race_metrics['FormScore'] = race_metrics['RecentWinRate'] / 100.0
        race_metrics['WeightedScore'] = (race_metrics['PaceScore'] * 0.7) + (race_metrics['FormScore'] * 0.3)

        # Filter to high confidence (0.80+)
        high_conf = race_metrics[race_metrics['WeightedScore'] >= 0.80].copy()

        if len(high_conf) > 0:
            high_conf['ImpliedProb'] = 1.0 / high_conf['StartingPrice']
            high_conf['ActualWinner'] = high_conf['Position'] == '1'
            high_conf['TrackName'] = 'Unknown'  # Will be populated from elsewhere if needed
            all_bets.append(high_conf)
            total_processed += len(high_conf)

progress(f"\nDone! Total high-confidence runners: {total_processed}")

if all_bets:
    combined = pd.concat(all_bets, ignore_index=True)

    progress("\n" + "="*120)
    progress("MISPRICE ANALYSIS: 70% Pace + 30% Form @ 0.80+ | $1.50-$3.00 Odds")
    progress("="*120)

    # Overall statistics
    total_runners = len(combined)
    winners = combined['ActualWinner'].astype(int).sum()
    overall_strike = (winners / total_runners) * 100
    overall_implied = combined['ImpliedProb'].mean() * 100
    overall_edge = overall_strike - overall_implied

    progress(f"\nOverall High-Confidence Dogs ($1.50-$3.00):", indent=1)
    progress(f"  Total runners: {total_runners}", indent=1)
    progress(f"  Actual wins: {winners}", indent=1)
    progress(f"  Actual strike rate: {overall_strike:.1f}%", indent=1)
    progress(f"  Market implied win%: {overall_implied:.1f}%", indent=1)
    progress(f"  Edge vs market: {overall_edge:+.1f}%", indent=1)

    # Find biggest misprice opportunities
    combined['MispriceBet'] = combined['WeightedScore'] - combined['ImpliedProb']
    combined['ROIPerBet'] = (combined['ActualWinner'].astype(float) * combined['StartingPrice'] - 1) / 1.0

    progress(f"\n" + "="*120)
    progress("TOP 20 BIGGEST MISPRICE OPPORTUNITIES (where model score >> market implied %)")
    progress("="*120)
    progress(f"\n{'Dog':20} | {'Track':15} | {'Dist':5} | {'Odds':6} | {'Score':6} | {'Market':8} | {'Result':5} | {'Edge':+6}", indent=1)
    progress("-" * 120, indent=1)

    top_misprice = combined.nlargest(20, 'MispriceBet')

    for _, dog in top_misprice.iterrows():
        result = "WIN" if dog['ActualWinner'] else "LOSS"
        track_name = dog.get('TrackName', 'Unknown')
        progress(f"{dog['GreyhoundName']:20} | {track_name:15} | {int(dog['Distance']):5} | ${dog['StartingPrice']:5.2f} | {dog['WeightedScore']:.3f} | {dog['ImpliedProb']*100:7.1f}% | {result:5} | {dog['MispriceBet']:+.3f}", indent=1)

    # Analyze by misprice level
    progress(f"\n" + "="*120)
    progress("MISPRICE LEVEL ANALYSIS")
    progress("="*120)

    bands = [
        (0.00, 0.05, "Barely misprice (0-5%)"),
        (0.05, 0.10, "Small misprice (5-10%)"),
        (0.10, 0.15, "Medium misprice (10-15%)"),
        (0.15, 1.00, "Large misprice (15%+)"),
    ]

    for band_min, band_max, label in bands:
        band = combined[(combined['MispriceBet'] >= band_min) & (combined['MispriceBet'] < band_max)]
        if len(band) > 0:
            band_strike = (band['ActualWinner'].astype(int).sum() / len(band)) * 100
            band_implied = band['ImpliedProb'].mean() * 100
            band_avg_odds = band['StartingPrice'].mean()
            band_roi = ((band['ActualWinner'].astype(int) * band['StartingPrice']).sum() - len(band)) / len(band) * 100
            
            progress(f"  {label:30} | {len(band):4} dogs | Strike {band_strike:5.1f}% vs Implied {band_implied:5.1f}% | Avg Odds ${band_avg_odds:.2f} | ROI {band_roi:+.1f}%", indent=1)

    # Best tracks
    progress(f"\n" + "="*120)
    progress("TRACK ANALYSIS (High-Confidence Dogs)")
    progress("="*120)

    if 'TrackName' in combined.columns and combined['TrackName'].notna().any():
        track_stats = combined.groupby('TrackName').agg({
            'ActualWinner': ['sum', 'count'],
            'StartingPrice': 'mean'
        }).round(3)
        
        track_stats.columns = ['Wins', 'Total', 'AvgOdds']
        track_stats['Strike%'] = (track_stats['Wins'] / track_stats['Total'] * 100).round(1)
        track_stats = track_stats[track_stats['Total'] >= 5].sort_values('Strike%', ascending=False)
        
        for track, row in track_stats.head(10).iterrows():
            progress(f"  {track:25} | {int(row['Total']):3} runners | {row['Strike%']:5.1f}% strike", indent=1)
    else:
        progress("  Track data not available", indent=1)

    # Best distances
    progress(f"\n" + "="*120)
    progress("DISTANCE ANALYSIS (High-Confidence Dogs)")
    progress("="*120)

    distance_stats = combined.groupby('Distance').agg({
        'ActualWinner': ['sum', 'count'],
        'StartingPrice': 'mean'
    }).round(3)
    
    distance_stats.columns = ['Wins', 'Total', 'AvgOdds']
    distance_stats['Strike%'] = (distance_stats['Wins'] / distance_stats['Total'] * 100).round(1)
    distance_stats = distance_stats[distance_stats['Total'] >= 5].sort_values('Strike%', ascending=False)
    
    for dist, row in distance_stats.iterrows():
        progress(f"  {int(dist)}m: {int(row['Total']):3} runners | {row['Strike%']:5.1f}% strike", indent=1)

    progress("\n" + "="*120)

else:
    progress("No high-confidence dogs found in backtest period")

conn.close()
