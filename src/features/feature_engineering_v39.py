"""Legacy feature engineering (V39) - DEPRECATED.

⚠️ Older version kept for reference. Use feature_engineering.py instead.
"""

import pandas as pd
import numpy as np
import re

class FeatureEngineerV39:
    """
    V39: Topaz-Enhanced Feature Engineering.
    Incorporates 'Useful' stats: RunHomeTime, PIR, Detailed Splits.
    """
    
    def __init__(self):
        pass
        
    def calculate_features(self, df):
        """
        Apply V39 feature engineering with Betfair 'Killer Features'.
        """
        # Ensure numeric types
        numeric_cols = ['FinishTime', 'TopazSplit1', 'TopazSplit2', 'Split', 'Box', 'Distance', 'Position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'MeetingDate' in df.columns:
            df['MeetingDate'] = pd.to_datetime(df['MeetingDate']).dt.tz_localize(None)
        if 'DateWhelped' in df.columns:
            df['DateWhelped'] = pd.to_datetime(df['DateWhelped'], errors='coerce').dt.tz_localize(None)

        # -------------------------------------------------------------------------
        # 1. Dog Age
        # -------------------------------------------------------------------------
        if 'DateWhelped' in df.columns and 'MeetingDate' in df.columns:
            df['DogAgeDays'] = (df['MeetingDate'] - df['DateWhelped']).dt.days
        else:
            df['DogAgeDays'] = np.nan

        # -------------------------------------------------------------------------
        # 2. Run Home Time (Strength) - Existing Logic
        # -------------------------------------------------------------------------
        df['RunHomeTime'] = np.nan
        mask_s2 = df['TopazSplit2'].notna() & (df['FinishTime'] > df['TopazSplit2'])
        df.loc[mask_s2, 'RunHomeTime'] = df.loc[mask_s2, 'FinishTime'] - df.loc[mask_s2, 'TopazSplit2']
        
        mask_s1_only = (~mask_s2) & df['Split'].notna() & (df['FinishTime'] > df['Split'])
        df.loc[mask_s1_only, 'RunHomeTime'] = df.loc[mask_s1_only, 'FinishTime'] - df.loc[mask_s1_only, 'Split']
        
        # -------------------------------------------------------------------------
        # 3. Finishing Place Movement (PIR)
        # -------------------------------------------------------------------------
        def calc_place_movement(row):
            pir = str(row.get('TopazPIR', ''))
            place = row.get('Position')
            if pd.isna(place) or not pir: return np.nan
            
            # Clean PIR (remove checks like 'C', 'M')
            pir = re.sub(r'[^0-9]', '', pir)
            if len(pir) < 2: return np.nan # Need at least 2 points
            
            # 2nd last digit usually represents position before home turn
            try:
                pos_turn = int(pir[-2]) 
                return pos_turn - place # Positive = Improved (Passed dogs), Negative = Faded
            except:
                return np.nan

        df['PlaceMovement'] = df.apply(calc_place_movement, axis=1)

        # -------------------------------------------------------------------------
        # 4. Vacant Box Analysis
        # -------------------------------------------------------------------------
        # Need to know if L/R boxes are empty in THIS race.
        # This requires grouping by RaceID.
        df['HasVacantBox'] = 0
        
        # Create mapping of RaceID+Box -> IsOccupied
        # Assuming df contains ALL runners in the races we are processing
        df['Occupied'] = 1
        
        # Optimization: Use sets for fast lookup
        occupied_set = set(zip(df['RaceID'], df['Box']))
        
        def check_vacant(row):
            race = row['RaceID']
            box = row['Box']
            if pd.isna(box) or box not in [1,2,3,4,5,6,7,8]: return 0
            
            # Check Left
            if box > 1:
                left_occ = (race, box-1) in occupied_set
            else:
                left_occ = True # Wall is "occupied" effectively
                
            # Check Right
            if box < 8:
                right_occ = (race, box+1) in occupied_set
            else:
                right_occ = True
                
            return 1 if (not left_occ or not right_occ) else 0

        df['HasVacantBox'] = df.apply(check_vacant, axis=1)

        # -------------------------------------------------------------------------
        # 5. Last 5 Win % (Form Parsing)
        # -------------------------------------------------------------------------
        def parse_form_win_pct(form_str):
            if not isinstance(form_str, str): return 0.0
            # Extract digits
            finishes = re.findall(r'\d', form_str)
            if not finishes: return 0.0
            
            wins = sum(1 for x in finishes if x == '1')
            return wins / len(finishes)

        if 'Form' in df.columns:
            df['Last5WinPct'] = df['Form'].apply(parse_form_win_pct)
        
        return df

    def enrich_with_rolling_stats(self, df):
        """
        Compute rolling averages and Track/Dist Norms.
        """
        # Ensure sorted
        df = df.sort_values(['GreyhoundID', 'MeetingDate'])
        
        # -------------------------------------------------------------------------
        # 6. Run Time Normalization (Speed Index) - LEAKAGE FREE VERSION
        # -------------------------------------------------------------------------
        # Calculate daily median winning times per Track/Distance
        winners = df[df['Position'] == 1].copy()
        
        if not winners.empty:
            # 1. Get Daily Median Winning Time for each Track/Dist
            daily_stats = winners.groupby(['TrackName', 'Distance', 'MeetingDate'])['FinishTime'].median().reset_index()
            daily_stats.rename(columns={'FinishTime': 'DailyWinTime'}, inplace=True)
            
            # 2. Calculate Rolling Median (Previous 365 Days)
            # We sort by date, then group by Track/Dist
            daily_stats = daily_stats.sort_values('MeetingDate')
            
            # Define a custom rolling function that SHIFTS first to avoid using current day's data
            def calc_rolling_benchmark(group):
                # Set date as index for time-based rolling
                g = group.set_index('MeetingDate')
                # Shift 1 to exclude current day, then roll 365 days
                # Note: shift(1) on a DataFrame/Series shifts by POSITION, not time.
                # Since we have one row per day (due to groupby checks), shift(1) moves to previous available day.
                # However, if there are gaps, we must be careful.
                # Better: Use closed='left' if pandas supports it, or manual shift.
                # Standard approach: Shift the values, then roll.
                
                # We need to reindex to ensure continuous days if we want true 365D? 
                # Or just rolling('365D') on observed days is fine.
                # Taking the observed days:
                return g['DailyWinTime'].shift(1).rolling('365D', min_periods=5).median()

            # Apply rolling
            benchmarks = daily_stats.groupby(['TrackName', 'Distance']).apply(calc_rolling_benchmark)
            
            # The result 'benchmarks' is a Series with MultiIndex (Track, Dist, MeetingDate) -> Value
            benchmarks.name = 'TrackDistMedian'
            benchmarks = benchmarks.reset_index()
            
            # 3. Merge back to main DF
            # This attaches the PREVIOUS year's median to the current row's MeetingDate
            df = pd.merge(df, benchmarks, on=['TrackName', 'Distance', 'MeetingDate'], how='left')
            
            # Forward fill to ensure inference days (no winner yet) get the latest benchmark
            # Sort by date just in case
            df = df.sort_values('MeetingDate')
            df['TrackDistMedian'] = df.groupby(['TrackName', 'Distance'])['TrackDistMedian'].ffill()
            
            # Forward fill benchmarks for days where there were runs but no winners (rare/impossible?) 
            # OR simple ffill within group to propagating the last known benchmark to current day
            # If today has races but no 'winner' explicitly calculated yet (inference time), 
            # we might rely on the last known value. 
            # For Training (history), the merge works for days that had winners.
            # But wait, 'daily_stats' only exists for days that HAD races. 
            # If 'df' has races, 'daily_stats' has entries.
            # So the merge should hit. 
            # IF the rolling returned NaN (first 5 races), we get NaN.
            
            # Fill initial NaNs with global average? Or leave as NaN?
            # Leave as NaN to be safe, or fill with an expandning mean if critical.
            # Let's fill with expanding mean as fallback to prevent data loss? 
            # Or just drop initial rows.
            
            # Calc RunTimeNorm (For history only - Do NOT use as direct feature for current race)
            df['RunTimeNorm'] = df['TrackDistMedian'] / df['FinishTime']
            df['SpeedIndex'] = df['TrackDistMedian'] / df['Distance'] 
            
        else:
             df['RunTimeNorm'] = np.nan
             df['SpeedIndex'] = np.nan
            
        # -------------------------------------------------------------------------
        # Rolling Lag Stats (The ACTUAL Features)
        # -------------------------------------------------------------------------
        windows = [1, 3] # Short term form
        
        # Sort for rolling
        df = df.sort_values(['GreyhoundID', 'MeetingDate'])
        grouped = df.groupby('GreyhoundID')
        
        for w in windows:
            # Shift 1 to exclude current race result
            
            # Run Home Strength
            df[f'RunHome_Lag{w}'] = grouped['RunHomeTime'].shift(1).rolling(w, min_periods=1).mean()
            
            # Split Speed
            df[f'Split_Lag{w}'] = grouped['Split'].shift(1).rolling(w, min_periods=1).mean()
            
            # Place Movement (Run Style)
            df[f'PlaceMove_Lag{w}'] = grouped['PlaceMovement'].shift(1).rolling(w, min_periods=1).mean()
            
            # Run Time Performance (The Killer Feature - Lagged)
            df[f'RunTimeNorm_Lag{w}'] = grouped['RunTimeNorm'].shift(1).rolling(w, min_periods=1).mean()
            
            # Speed Index
            df[f'SpeedIndex_Lag{w}'] = grouped['SpeedIndex'].shift(1).rolling(w, min_periods=1).mean()

        return df
