import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'paper_trading.db'
RACING_DB_PATH = 'greyhound_racing.db'

class PaperTradingManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize the paper trading database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PaperBets (
            BetID INTEGER PRIMARY KEY AUTOINCREMENT,
            BetDate TEXT,
            GreyhoundName TEXT,
            RaceNumber INTEGER,
            TrackName TEXT,
            Strategy TEXT,
            Odds REAL,
            Stake REAL,
            Status TEXT DEFAULT 'OPEN', -- OPEN, WIN, LOSS, VOID
            Return REAL DEFAULT 0,
            Profit REAL DEFAULT 0,
            ResultTime TEXT
        )
        ''')
        
        conn.commit()
        conn.close()

    def place_bet(self, bet_data):
        """
        Place a new paper bet.
        bet_data: dict with keys (GreyhoundName, RaceNumber, TrackName, Strategy, Odds, Stake)
        Returns True if placed, False if duplicate.
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for duplicates (same dog, race, track, open status)
        # using strictly normalized comparisons to be safe
        dog_norm = str(bet_data['GreyhoundName']).strip().upper()
        race_num = int(bet_data['RaceNumber'])
        track_norm = str(bet_data['TrackName']).strip().upper()
        
        cursor.execute('''
            SELECT COUNT(*) FROM PaperBets 
            WHERE UPPER(TRIM(GreyhoundName)) = ? 
              AND RaceNumber = ? 
              AND UPPER(TRIM(TrackName)) = ?
              AND Status = 'OPEN'
        ''', (dog_norm, race_num, track_norm))
        
        count = cursor.fetchone()[0]
        if count > 0:
            conn.close()
            print(f"[INFO] Skipped duplicate bet: {dog_norm} {race_num} {track_norm}")
            return False

        cursor.execute('''
        INSERT INTO PaperBets (BetDate, GreyhoundName, RaceNumber, TrackName, Strategy, Odds, Stake, Status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            bet_data['GreyhoundName'],
            bet_data['RaceNumber'],
            bet_data['TrackName'],
            bet_data['Strategy'],
            bet_data['Odds'],
            bet_data['Stake']
        ))
        
        conn.commit()
        conn.close()
        return True

    def get_active_bets(self):
        """Get all open bets"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM PaperBets WHERE Status='OPEN' ORDER BY BetDate DESC", conn)
        conn.close()
        return df

    def get_bet_history(self):
        """Get all settled bets"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM PaperBets WHERE Status!='OPEN' ORDER BY BetDate DESC", conn)
        conn.close()
        return df

    def get_stats(self):
        """Calculate performance stats"""
        df = self.get_bet_history()
        if len(df) == 0:
            return {'bets': 0, 'wins': 0, 'strike_rate': 0, 'profit': 0, 'roi': 0}
            
        wins = len(df[df['Status'] == 'WIN'])
        total_bets = len(df)
        profit = df['Profit'].sum()
        total_stake = df['Stake'].sum()
        
        return {
            'bets': total_bets,
            'wins': wins,
            'strike_rate': (wins / total_bets) * 100,
            'profit': profit,
            'roi': (profit / total_stake) * 100 if total_stake > 0 else 0
        }

    def reconcile_bets(self):
        """Check racing DB for results of open bets"""
        # 1. Get Open Bets
        active_bets = self.get_active_bets()
        if len(active_bets) == 0:
            return 0

        # 2. Connect to Racing DB to find results
        race_conn = sqlite3.connect(RACING_DB_PATH)
        
        updates = 0
        paper_conn = sqlite3.connect(DB_PATH)
        paper_cursor = paper_conn.cursor()

        for _, bet in active_bets.iterrows():
            # Query matching dog entry, including BSP
            query = """
            SELECT ge.Position, ge.StartingPrice, ge.BSP
            FROM GreyhoundEntries ge
            JOIN Races r ON ge.RaceID = r.RaceID
            JOIN RaceMeetings rm ON r.MeetingID = rm.MeetingID
            JOIN Tracks t ON rm.TrackID = t.TrackID
            JOIN Greyhounds g ON ge.GreyhoundID = g.GreyhoundID
            WHERE g.GreyhoundName = ?
              AND t.TrackName = ?
              AND r.RaceNumber = ?
              AND ge.Position NOT IN ('', 'SCR')
              -- Look for races in last 48 hours to avoid old matches
              AND rm.MeetingDate >= date('now', '-2 days') 
            """
            
            try:
                df_res = pd.read_sql_query(query, race_conn, params=(bet['GreyhoundName'], bet['TrackName'], bet['RaceNumber']))
                
                if not df_res.empty:
                    pos_str = str(df_res.iloc[0]['Position'])
                    
                    # Determine odds to use for settlement
                    # Prefer BSP if available and non-zero
                    settle_price = bet['Odds'] # Default to taken price
                    bsp = df_res.iloc[0]['BSP']
                    
                    # If bet was placed with SP/BSP in mind (Odds=0 or similar indicator), or user wants BSP settlement
                    # For now, we stick to Taken Odds unless user entered 0/None?
                    # But user said "fill out columns other than odd obviously" -> implying we might settle at SP/BSP.
                    # Let's assume if Bet Odds is 0 or empty, we use BSP or SP.
                    if settle_price is None or settle_price <= 1.0:
                         if bsp and bsp > 1.0:
                             settle_price = bsp
                         else:
                             # Try parsing SP string e.g. "$2.50"
                             sp_str = str(df_res.iloc[0]['StartingPrice']).replace('$', '').replace('F', '')
                             try:
                                 settle_price = float(sp_str)
                             except:
                                 settle_price = 0

                    if settle_price <= 1.0:
                        print(f"Skipping bet {bet['BetID']}: No valid settlement price.")
                        continue

                    # Strategy Logic
                    strategy = str(bet['Strategy']).upper()
                    is_lay = 'LAY' in strategy or 'LAYER' in strategy

                    new_status = 'OPEN'
                    profit = 0
                    ret = 0
                    
                    # Clean position (handle dead heats?) - assume simple for now
                    if pos_str == '1':
                        # Dog WON
                        if is_lay:
                            new_status = 'LOSS'
                            # Loss = Stake * (Odds - 1)
                            # Assumption: Stake is "Backer's Stake" (Volume) we want to win
                            profit = -1 * bet['Stake'] * (settle_price - 1)
                            ret = 0
                        else:
                            new_status = 'WIN'
                            ret = bet['Stake'] * settle_price
                            profit = ret - bet['Stake']
                    else:
                        # Dog LOST (2nd, 3rd, ...)
                        if is_lay:
                            new_status = 'WIN'
                            # Valid Win: Profit = Stake * (1 - Comm)
                            COMMISSION = 0.05
                            profit = bet['Stake'] * (1 - COMMISSION)
                            ret = bet['Stake'] + profit
                        else:
                            new_status = 'LOSS'
                            profit = -bet['Stake']
                            ret = 0
                    
                    if new_status != 'OPEN':
                        paper_cursor.execute('''
                            UPDATE PaperBets 
                            SET Status=?, Return=?, Profit=?, ResultTime=?, Odds=?
                            WHERE BetID=?
                        ''', (new_status, ret, profit, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), settle_price, bet['BetID']))
                        updates += 1
                    
            except Exception as e:
                print(f"Error reconciling bet {bet['BetID']}: {e}")

        paper_conn.commit()
        paper_conn.close()
        race_conn.close()
        
        return updates

    def delete_bet(self, bet_id):
        """Delete a bet by ID"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM PaperBets WHERE BetID = ?", (bet_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0
