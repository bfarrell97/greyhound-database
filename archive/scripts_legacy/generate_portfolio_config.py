import pandas as pd

def generate_config():
    df = pd.read_csv('track_analysis.csv')
    
    # Filter Profitable Tracks
    profitable = df[df['ROI'] > 0].copy()
    
    print("PORTFOLIO_CONFIG = {")
    for _, row in profitable.iterrows():
        track = row['Track']
        strat = row['Strategy']
        # We need to map 'Strategy' column to our internal names
        # In analysis: 'Dominator', 'Swooper'
        # In backtest logic, we'll use these keys.
        
        print(f"    '{track}': ['{strat}'],")
        
    print("}")
    
    # Also print summary for log
    print("\nSummary of Profitable Tracks:")
    print(profitable[['Track', 'Strategy', 'ROI', 'Bets']].sort_values('ROI', ascending=False).to_string())

if __name__ == "__main__":
    generate_config()
