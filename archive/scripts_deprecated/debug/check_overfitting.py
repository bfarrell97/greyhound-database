import pandas as pd
import numpy as np

def check_overfitting():
    print("CHECKING FOR OVERFITTING (Robustness Test)")
    print("=" * 60)
    
    csv_file = 'results/backtest_longterm_top3.csv'
    try:
        df = pd.read_csv(csv_file)
        df['MeetingDate'] = pd.to_datetime(df['MeetingDate'])
        df['Year'] = df['MeetingDate'].dt.year
        
        # Ensure correct types
        df['RunningPrize'] = pd.to_numeric(df['RunningPrize'])
        df['PaceRank'] = pd.to_numeric(df['PaceRank'])
        df['Profit'] = pd.to_numeric(df['Profit'])
        df['Stake'] = pd.to_numeric(df['Stake'])
        
        print(f"Loaded {len(df)} bets.")
        
        # 1. Yearly Stability
        print("\n1. YEARLY STABILITY (Consistency Check)")
        print("-" * 50)
        yearly = df.groupby('Year').agg({
            'Profit': 'sum',
            'Stake': 'sum',
            'RunningPrize': 'count'
        }).rename(columns={'RunningPrize': 'Bets'})
        
        yearly['ROI'] = (yearly['Profit'] / yearly['Stake']) * 100
        print(yearly[['Bets', 'Profit', 'ROI']])
        
        # Check variance
        rois = yearly['ROI']
        if rois.min() > 0:
            print(">> PASS: Profitable in ALL years.")
        else:
            print(">> WARNING: Unprofitable years detected.")

        # 2. Parameter Sensitivity (Prize Threshold)
        print("\n2. SENSITIVITY: Prize Money Threshold (Base: $30k)")
        print("-" * 50)
        thresholds = [20000, 25000, 30000, 35000, 40000]
        
        for t in thresholds:
            # Filter subset
            # Note: The CSV only contains bets with RunningPrize >= 30000 
            # (because the backtest script filtered them).
            # So we can only test HIGHER thresholds (Restriction).
            # Wait, if I filtered >= 30k in script, I can't test 20k.
            # I can only test > 30k.
            # But checking >30k sensitivity is still valid.
            
            if t < 30000:
                continue
                
            subset = df[df['RunningPrize'] >= t]
            if len(subset) == 0:
                print(f"  ${t/1000}k+: No bets")
                continue
                
            roi = (subset['Profit'].sum() / subset['Stake'].sum()) * 100
            print(f"  ${t/1000}k+: ROI {roi:+.1f}% ({len(subset)} bets)")
            
        print(">> Analysis: If ROI remains positive/stable at higher thresholds, it is robust.")

        # 3. Box Bias
        print("\n3. BOX BIAS (Is it just betting Box 1/8?)")
        print("-" * 50)
        box_stats = df.groupby('Box').agg({
            'Profit': 'sum',
            'Stake': 'sum',
            'MeetingDate': 'count'
        }).rename(columns={'MeetingDate': 'Bets'})
        
        box_stats['ROI'] = (box_stats['Profit'] / box_stats['Stake']) * 100
        print(box_stats[['Bets', 'ROI']])
        
        # Check if spread is reasonable
        profitable_boxes = (box_stats['ROI'] > 0).sum()
        print(f">> Profitable Boxes: {profitable_boxes}/8")
        if profitable_boxes >= 6:
            print(">> PASS: Strategy works across most boxes.")
        else:
            print(">> WARNING: Strategy relies heavily on specific boxes.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_overfitting()
