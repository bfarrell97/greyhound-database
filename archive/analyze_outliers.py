"""
OUTLIER DETECTION: Identify and exclude extreme benchmark values
Prevents single anomalous races from skewing pace calculations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = 'greyhound_racing.db'

def progress(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

progress("="*120)
progress("OUTLIER ANALYSIS: Detecting extreme benchmark values")
progress("="*120)

conn = sqlite3.connect(DB_PATH)

# Get all benchmark values with dog and race info
query = """
SELECT 
    g.GreyhoundID,
    g.GreyhoundName,
    rm.MeetingDate,
    t.TrackName,
    r.Distance,
    ge.Position,
    ge.FinishTime,
    ge.FinishTimeBenchmarkLengths,
    rm.MeetingAvgBenchmarkLengths,
    (ge.FinishTimeBenchmarkLengths + rm.MeetingAvgBenchmarkLengths) as TotalFinishBench
FROM GreyhoundEntries ge
JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
JOIN Races r ON ge.RaceID = r.RaceID
JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
JOIN Tracks t ON rm.TrackID = t.TrackID
WHERE ge.FinishTimeBenchmarkLengths IS NOT NULL
  AND rm.MeetingAvgBenchmarkLengths IS NOT NULL
  AND ge.Position IS NOT NULL
  AND ge.Position NOT IN ('DNF', 'SCR')
  AND rm.MeetingDate >= '2025-01-01'
ORDER BY ABS(ge.FinishTimeBenchmarkLengths) DESC
"""

progress("\nLoading all benchmark data...")
df = pd.read_sql_query(query, conn)
conn.close()

progress(f"Loaded {len(df):,} race entries")

# Calculate statistics
df['TotalFinishBench'] = pd.to_numeric(df['TotalFinishBench'], errors='coerce')
df['FinishTimeBenchmarkLengths'] = pd.to_numeric(df['FinishTimeBenchmarkLengths'], errors='coerce')

# Calculate percentiles and outlier boundaries
q75, q25 = df['FinishTimeBenchmarkLengths'].quantile([0.75, 0.25])
iqr = q75 - q25
lower_bound = q25 - (1.5 * iqr)
upper_bound = q75 + (1.5 * iqr)

# Identify outliers
df['IsOutlier'] = (df['FinishTimeBenchmarkLengths'] < lower_bound) | (df['FinishTimeBenchmarkLengths'] > upper_bound)

progress(f"\n{'='*120}")
progress("OUTLIER DETECTION RESULTS")
progress(f"{'='*120}")
progress(f"Q1 (25th percentile): {q25:.2f}")
progress(f"Q3 (75th percentile): {q75:.2f}")
progress(f"IQR: {iqr:.2f}")
progress(f"Lower Boundary: {lower_bound:.2f}")
progress(f"Upper Boundary: {upper_bound:.2f}")
progress(f"\nTotal outliers detected: {df['IsOutlier'].sum():,} out of {len(df):,} ({df['IsOutlier'].sum()/len(df)*100:.1f}%)")

# Show extreme outliers
progress(f"\n{'='*120}")
progress("TOP 20 EXTREME OUTLIERS (Most Extreme)")
progress(f"{'='*120}\n")

df['AbsBench'] = df['FinishTimeBenchmarkLengths'].abs()
extremes = df.nlargest(20, 'AbsBench')
display_df = extremes[[
    'GreyhoundName', 'MeetingDate', 'TrackName', 'Distance', 'Position', 
    'FinishTime', 'FinishTimeBenchmarkLengths', 'TotalFinishBench'
]].copy()

display_df.columns = ['Dog', 'Date', 'Track', 'Distance', 'Pos', 'FinishTime', 'FTBench', 'TotalBench']
print(display_df.to_string(index=False))

# Analyze by dog - which dogs are affected by outliers
progress(f"\n{'='*120}")
progress("DOGS MOST AFFECTED BY OUTLIERS")
progress(f"{'='*120}\n")

outlier_dogs = df[df['IsOutlier']].groupby('GreyhoundName').agg({
    'IsOutlier': 'sum',
    'GreyhoundName': 'count',
    'FinishTimeBenchmarkLengths': ['min', 'max']
}).round(2)

outlier_dogs.columns = ['OutlierCount', 'TotalRaces', 'MinBench', 'MaxBench']
outlier_dogs = outlier_dogs[outlier_dogs['OutlierCount'] > 0].sort_values('OutlierCount', ascending=False)

print(outlier_dogs.head(20).to_string())

# Proposed solution: Show impact of excluding outliers
progress(f"\n{'='*120}")
progress("IMPACT OF EXCLUDING OUTLIERS")
progress(f"{'='*120}\n")

# Example: INFRARED GEM
ig_data = df[df['GreyhoundName'] == 'INFRARED GEM'].sort_values('MeetingDate', ascending=False).head(5)
progress("INFRARED GEM - Last 5 races:")
print(ig_data[['MeetingDate', 'Distance', 'Position', 'FinishTime', 'FinishTimeBenchmarkLengths', 'IsOutlier']].to_string(index=False))

ig_with_outliers = ig_data['FinishTimeBenchmarkLengths'].mean()
ig_without_outliers = ig_data[~ig_data['IsOutlier']]['FinishTimeBenchmarkLengths'].mean()

progress(f"\nWith outliers: {ig_with_outliers:.2f}")
progress(f"Without outliers: {ig_without_outliers:.2f}")
progress(f"Difference: {ig_with_outliers - ig_without_outliers:.2f} lengths")

progress(f"\n{'='*120}")
progress("RECOMMENDATION")
progress(f"{'='*120}")
progress(f"\nExclude benchmarks outside the IQR bounds:")
progress(f"  - LOWER: {lower_bound:.2f}")
progress(f"  - UPPER: {upper_bound:.2f}")
progress(f"\nThis removes {df['IsOutlier'].sum():,} anomalous races ({df['IsOutlier'].sum()/len(df)*100:.1f}%)")
progress(f"while preserving {(~df['IsOutlier']).sum():,} legitimate race performances")

# Export outlier-filtered data for use in production
progress(f"\n{'='*120}")
progress("CREATING OUTLIER-FILTERED DATASET")
progress(f"{'='*120}\n")

df_clean = df[~df['IsOutlier']].copy()
progress(f"Original dataset: {len(df):,} races")
progress(f"After outlier removal: {len(df_clean):,} races")
progress(f"Data quality improvement: {(df['IsOutlier'].sum()/len(df)*100):.1f}% anomalies removed")

# Show pace statistics before/after
progress(f"\n{'='*120}")
progress("PACE STATISTICS: BEFORE vs AFTER OUTLIER REMOVAL")
progress(f"{'='*120}\n")

stats_comparison = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
    'With Outliers': [
        df['FinishTimeBenchmarkLengths'].mean(),
        df['FinishTimeBenchmarkLengths'].median(),
        df['FinishTimeBenchmarkLengths'].std(),
        df['FinishTimeBenchmarkLengths'].min(),
        df['FinishTimeBenchmarkLengths'].max(),
        df['FinishTimeBenchmarkLengths'].quantile(0.25),
        df['FinishTimeBenchmarkLengths'].quantile(0.75)
    ],
    'Without Outliers': [
        df_clean['FinishTimeBenchmarkLengths'].mean(),
        df_clean['FinishTimeBenchmarkLengths'].median(),
        df_clean['FinishTimeBenchmarkLengths'].std(),
        df_clean['FinishTimeBenchmarkLengths'].min(),
        df_clean['FinishTimeBenchmarkLengths'].max(),
        df_clean['FinishTimeBenchmarkLengths'].quantile(0.25),
        df_clean['FinishTimeBenchmarkLengths'].quantile(0.75)
    ]
})

stats_comparison = stats_comparison.round(3)
print(stats_comparison.to_string(index=False))

# Impact on dogs with outliers
progress(f"\n{'='*120}")
progress("IMPACT ON SPECIFIC DOGS - PACE SCORES WITH/WITHOUT OUTLIERS")
progress(f"{'='*120}\n")

# Get dogs with outliers in their history
dogs_with_outliers = df[df['IsOutlier']]['GreyhoundName'].unique()[:10]

impact_data = []
for dog in dogs_with_outliers:
    dog_df_all = df[df['GreyhoundName'] == dog].sort_values('MeetingDate', ascending=False).head(5)
    dog_df_clean = dog_df_all[~dog_df_all['IsOutlier']]
    
    if len(dog_df_all) >= 3:  # Only if sufficient data
        pace_with = dog_df_all['FinishTimeBenchmarkLengths'].mean()
        pace_without = dog_df_clean['FinishTimeBenchmarkLengths'].mean() if len(dog_df_clean) > 0 else 0
        outlier_count = dog_df_all['IsOutlier'].sum()
        
        impact_data.append({
            'Dog': dog,
            'OutliersInLast5': int(outlier_count),
            'PaceWithOutliers': round(pace_with, 2),
            'PaceWithoutOutliers': round(pace_without, 2),
            'Difference': round(pace_with - pace_without, 2)
        })

impact_df = pd.DataFrame(impact_data).sort_values('Difference', ascending=False, key=abs)
print(impact_df.to_string(index=False))

# Export results to CSV for documentation
progress(f"\n{'='*120}")
progress("EXPORTING ANALYSIS RESULTS")
progress(f"{'='*120}\n")

# Export full outlier analysis
outlier_details = df[df['IsOutlier']][['GreyhoundName', 'MeetingDate', 'TrackName', 'Distance', 
                                        'Position', 'FinishTime', 'FinishTimeBenchmarkLengths']].copy()
outlier_details.columns = ['Dog', 'Date', 'Track', 'Distance', 'Position', 'FinishTime', 'Benchmark']
outlier_details = outlier_details.sort_values('Benchmark', key=abs, ascending=False)
outlier_details.to_csv('outliers_detected_2025.csv', index=False)
progress(f"[OK] Exported {len(outlier_details):,} outliers to outliers_detected_2025.csv")

# Export configuration for production
config_output = f"""
# Outlier Detection Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OUTLIER_LOWER_BOUND = {lower_bound:.2f}
OUTLIER_UPPER_BOUND = {upper_bound:.2f}
OUTLIER_IQR_MULTIPLIER = 1.5

# Statistics
TOTAL_RACES_ANALYZED = {len(df):,}
OUTLIERS_DETECTED = {df['IsOutlier'].sum():,}
OUTLIER_PERCENTAGE = {df['IsOutlier'].sum()/len(df)*100:.2f}%

# Usage in SQL:
# WHERE FinishTimeBenchmarkLengths BETWEEN {lower_bound:.2f} AND {upper_bound:.2f}
"""

with open('outlier_config.py', 'w') as f:
    f.write(config_output)

progress(f"[OK] Exported outlier configuration to outlier_config.py")

progress(f"\n{'='*120}")
progress("ANALYSIS COMPLETE")
progress(f"{'='*120}\n")
progress("Next steps:")
progress("1. Review outliers_detected_2025.csv for data quality issues")
progress("2. Update production queries to use outlier bounds")
progress("3. Re-validate model performance with clean data")
progress("4. Monitor for new outlier patterns in production data")
progress(f"\n{'='*120}\n")
progress(f"  - UPPER: {upper_bound:.2f}")
progress(f"\nThis removes {df['IsOutlier'].sum():,} anomalous races ({df['IsOutlier'].sum()/len(df)*100:.1f}%)")
progress(f"while preserving {(~df['IsOutlier']).sum():,} legitimate race performances")
progress(f"\n{'='*120}\n")
