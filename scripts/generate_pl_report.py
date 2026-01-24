import pandas as pd
import datetime
import os

def generate_report():
    input_file = "live_bets.csv"
    output_xlsx = "live_bets_summary.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure correct types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
    df['Stake'] = pd.to_numeric(df['Stake'], errors='coerce').fillna(0)
    
    # Filter for Matched/Settled only? 
    # User said "they were probably unmatched".
    # We should filter for Settled bets for the P/L calculation, 
    # but maybe show "Unmatched" count.
    # Actually, Profit is only non-zero for settled bets, so summing Profit works for all rows.
    # ROI = Profit / Turnover (Stake of SETTLED bets).
    # If Status is UNMATCHED, Stake is 0 effective? No, Stake is intended stake.
    # We should only sum Stake where Status is SETTLED/WIN/LOSS.
    
    settled_mask = df['Status'].isin(['SETTLED', 'WIN', 'LOSS']) | (df['Profit'] != 0)
    settled_df = df[settled_mask].copy()
    
    if settled_df.empty:
        print("No settled bets found to analyze.")
        return

    print(f"Analying {len(settled_df)} settled bets...")
    
    # --- HELPER --
    def get_stats(d):
        # Overall
        total_p = d['Profit'].sum()
        total_to = d['Stake'].sum()
        total_roi = (total_p / total_to * 100) if total_to > 0 else 0.0
        total_wins = len(d[d['Profit'] > 0])
        total_bets = len(d)
        total_sr = (total_wins / total_bets * 100) if total_bets > 0 else 0.0
        
        # BACK
        back_d = d[d['BetType'] == 'BACK']
        back_p = back_d['Profit'].sum()
        back_to = back_d['Stake'].sum()
        back_roi = (back_p / back_to * 100) if back_to > 0 else 0.0
        back_wins = len(back_d[back_d['Profit'] > 0])
        back_bets = len(back_d)
        back_sr = (back_wins / back_bets * 100) if back_bets > 0 else 0.0

        # LAY
        lay_d = d[d['BetType'] == 'LAY']
        lay_p = lay_d['Profit'].sum()
        lay_to = lay_d['Stake'].sum()
        lay_roi = (lay_p / lay_to * 100) if lay_to > 0 else 0.0
        lay_wins = len(lay_d[lay_d['Profit'] > 0])
        lay_bets = len(lay_d)
        lay_sr = (lay_wins / lay_bets * 100) if lay_bets > 0 else 0.0
        
        return pd.Series({
            'Bets': total_bets,
            'Wins': total_wins,
            'StrikeRate': f"{total_sr:.1f}%",
            'Turnover': f"${total_to:.2f}",
            'Profit': f"${total_p:.2f}",
            'ROI': f"{total_roi:.1f}%",
            
            'BACK_Bets': back_bets,
            'BACK_SR': f"{back_sr:.1f}%",
            'BACK_Profit': f"${back_p:.2f}",
            'BACK_ROI': f"{back_roi:.1f}%",
            
            'LAY_Bets': lay_bets,
            'LAY_SR': f"{lay_sr:.1f}%",
            'LAY_Profit': f"${lay_p:.2f}",
            'LAY_ROI': f"{lay_roi:.1f}%"
        })

    # 1. DAILY
    daily = settled_df.groupby(settled_df['Date'].dt.date).apply(get_stats)
    daily.index.name = 'Date'
    
    # 2. WEEKLY
    weekly = settled_df.groupby(settled_df['Date'].dt.to_period('W')).apply(get_stats)
    weekly.index.name = 'Week'
    
    # 3. MONTHLY
    monthly = settled_df.groupby(settled_df['Date'].dt.to_period('M')).apply(get_stats)
    monthly.index.name = 'Month'
    
    # 4. OVERALL
    overall = get_stats(settled_df).to_frame().T
    overall.index = ['Overall']

    # EXPORT TO EXCEL
    try:
        print(f"Writing to {output_xlsx}...")
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            # Raw Data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Summary Sheets
            daily.to_excel(writer, sheet_name='Daily Report')
            weekly.to_excel(writer, sheet_name='Weekly Report')
            monthly.to_excel(writer, sheet_name='Monthly Report')
            overall.to_excel(writer, sheet_name='Overall Summary')
            
        print("Success! Created Excel report.")
        
    except ImportError:
        print("Error: 'openpyxl' library missing. Falling back to CSVs.")
        daily.to_csv("report_daily.csv")
        weekly.to_csv("report_weekly.csv")
        monthly.to_csv("report_monthly.csv")
        overall.to_csv("report_overall.csv")
        print("Created separate CSV reports.")
        
    except Exception as e:
        print(f"Error writing Excel: {e}")

if __name__ == "__main__":
    generate_report()
