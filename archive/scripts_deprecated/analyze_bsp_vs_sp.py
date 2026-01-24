import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = 'greyhound_racing.db'

def analyze_bsp_vs_sp():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    
    # query: Get records where both SP and BSP exist
    # Clean the SP string in SQL or Python. Let's do it in Python for safety.
    query = """
    SELECT 
        StartingPrice, 
        BSP,
        Position,
        TrackName
    FROM GreyhoundEntries ge
    JOIN Races r ON ge.RaceID = r.RaceID
    JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
    JOIN Tracks t ON rm.TrackID = t.TrackID
    WHERE BSP IS NOT NULL 
      AND StartingPrice IS NOT NULL 
      AND StartingPrice != ''
    """
    
    print("Fetching data (All Runners to check market pricing)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No records found with both BSP and SP.")
        return

    print(f"Analyzing {len(df)} Winning bets...")

    # Data Cleaning
    def parse_price(p_str):
        if not isinstance(p_str, str): return p_str
        try:
            # Remove '$', 'F', spaces
            clean = p_str.replace('$', '').replace('F', '').replace(' ', '')
            return float(clean)
        except:
            return None

    df['SP_Float'] = df['StartingPrice'].apply(parse_price)
    df = df.dropna(subset=['SP_Float', 'BSP'])
    
    # Comparison Logic
    df['BSP_Net'] = df['BSP'] * 0.95 # Deduct 5% commission
    
    # 1. Direct Comparison Count
    df['BSP_Better'] = df['BSP_Net'] > df['SP_Float']
    bsp_better_count = df['BSP_Better'].sum()
    bsp_better_pct = (bsp_better_count / len(df)) * 100
    
    print("\n" + "="*50)
    print("WINNER PRICE COMPARISON")
    print("="*50)
    print(f"Total Winners Analyzed: {len(df)}")
    print(f"BSP (Net 5%) was better than SP in {bsp_better_count} cases ({bsp_better_pct:.1f}%)")
    
    # 2. Average Prices
    avg_sp = df['SP_Float'].mean()
    avg_bsp = df['BSP'].mean()
    avg_bsp_net = df['BSP_Net'].mean()
    
    print(f"\nAverage SP Price:     ${avg_sp:.2f}")
    print(f"Average Raw BSP:      ${avg_bsp:.2f}")
    print(f"Average Net BSP (5%): ${avg_bsp_net:.2f}")
    print(f"Difference (Net):     {((avg_bsp_net - avg_sp)/avg_sp)*100:+.1f}%")
    
    # 3. ROI Simulation (Flat Stake $1 on every winner)
    # Note: This is an idealized 'Crystal Ball' ROI (betting on winners only), 
    # but meaningful for Comparing the Payout Potential.
    profit_sp = df['SP_Float'].sum() - len(df) # Profit = Odds - 1 (Stake)
    profit_bsp = df['BSP_Net'].sum() - len(df)
    
    print("\n" + "="*50)
    print("PAYOUT SIMULATION (If you backed every winner)")
    print("="*50)
    print(f"Profit at SP:  ${profit_sp:.2f}")
    print(f"Profit at BSP: ${profit_bsp:.2f}")
    lift = profit_bsp - profit_sp
    print(f"BSP Advantage: ${lift:.2f} ({(lift/profit_sp)*100:+.1f}% more profit)")

    # 4. By Price Buckets
    print("\n" + "="*50)
    print("BREAKDOWN BY ODDS RANGE")
    print("="*50)
    bins = [0, 2, 5, 10, 20, 1000]
    labels = ['Fav (<$2)', 'Short ($2-5)', 'Mid ($5-10)', 'Long ($10-20)', 'Roughie (>$20)']
    df['OddsRange'] = pd.cut(df['SP_Float'], bins=bins, labels=labels)
    
    grouped = df.groupby('OddsRange').agg({
        'SP_Float': 'mean',
        'BSP_Net': 'mean',
        'BSP_Better': 'mean' # acts as percentage
    })
    
    print(f"{'Range':<15} {'Avg SP':<10} {'Avg BSP(Net)':<15} {'% BSP Better'}")
    for idx, row in grouped.iterrows():
        print(f"{idx:<15} ${row['SP_Float']:<9.2f} ${row['BSP_Net']:<14.2f} {row['BSP_Better']*100:.1f}%")

if __name__ == "__main__":
    analyze_bsp_vs_sp()
