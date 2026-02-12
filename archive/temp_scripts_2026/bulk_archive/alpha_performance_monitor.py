"""
Alpha Performance Monitor
Tracks ROI for V42/V43 Alpha bets in real-time.
Reads from LiveBets table and calculates running statistics.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "greyhound_racing.db"

def get_alpha_performance(days_back=30):
    """Calculate ROI for Alpha bets over the past N days."""
    conn = sqlite3.connect(DB_PATH)
    
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    query = f"""
    SELECT 
        lb.BetID, lb.RunnerName as Dog, lb.EventName as Track, lb.Side as BetType, 
        lb.StrategyUsed as Strategy,
        lb.Size as Stake, lb.Price as Odds, lb.Status, lb.Profit as PnL, lb.PlacedDate as PlacedAt
    FROM LiveBets lb
    WHERE lb.PlacedDate >= '{cutoff}'
    AND (lb.StrategyUsed LIKE '%ALPHA%' OR lb.StrategyUsed LIKE '%V42%' OR lb.StrategyUsed LIKE '%V43%')
    AND lb.Status = 'SETTLED'
    ORDER BY lb.PlacedDate DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No settled Alpha bets in the past {days_back} days.")
        return None
    
    print("="*70)
    print(f"ALPHA PERFORMANCE REPORT ({days_back} Days)")
    print("="*70)
    
    # Overall Stats
    total_bets = len(df)
    total_stake = df['Stake'].sum()
    total_pnl = df['PnL'].sum()
    overall_roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0
    
    print(f"\nOverall: {total_bets} bets | Stake: ${total_stake:.2f} | PnL: ${total_pnl:.2f} | ROI: {overall_roi:.2f}%")
    
    # By Bet Type
    print("\n--- By Bet Type ---")
    for bt in df['BetType'].unique():
        sub = df[df['BetType'] == bt]
        stake = sub['Stake'].sum()
        pnl = sub['PnL'].sum()
        roi = (pnl / stake * 100) if stake > 0 else 0
        wins = (sub['PnL'] > 0).sum()
        print(f"  {bt}: {len(sub)} bets | Win Rate: {wins/len(sub)*100:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
    
    # By Strategy
    print("\n--- By Strategy ---")
    for strat in df['Strategy'].unique():
        sub = df[df['Strategy'] == strat]
        stake = sub['Stake'].sum()
        pnl = sub['PnL'].sum()
        roi = (pnl / stake * 100) if stake > 0 else 0
        print(f"  {strat}: {len(sub)} bets | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")
    
    # Daily Breakdown
    print("\n--- Daily P&L ---")
    df['Date'] = pd.to_datetime(df['PlacedAt']).dt.date
    daily = df.groupby('Date').agg({'PnL': 'sum', 'Stake': 'sum', 'BetID': 'count'}).rename(columns={'BetID': 'Bets'})
    daily['ROI'] = daily['PnL'] / daily['Stake'] * 100
    daily = daily.sort_index(ascending=False).head(10)
    print(daily.to_string())
    
    print("="*70)
    
    return df


def log_alpha_bet(dog, track, bet_type, strategy, stake, odds, status='PENDING', pnl=0.0):
    """Log a new Alpha bet to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO LiveBets (Dog, Track, BetType, Strategy, Stake, Odds, Status, PnL, PlacedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (dog, track, bet_type, strategy, stake, odds, status, pnl, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    print(f"[LOGGED] {bet_type} {dog} @ {track} | ${stake:.2f} @ {odds:.2f}")


def settle_bet(bet_id, pnl):
    """Settle a bet with final PnL."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE LiveBets SET Status = 'SETTLED', PnL = ? WHERE BetID = ?", (pnl, bet_id))
    conn.commit()
    conn.close()
    print(f"[SETTLED] BetID {bet_id} | PnL: ${pnl:.2f}")


def export_report(days_back=30):
    """Export performance report to CSV."""
    df = get_alpha_performance(days_back)
    if df is not None:
        filename = f"alpha_performance_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nExported to {filename}")


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    get_alpha_performance(days)
