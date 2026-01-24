
import requests
import json
import threading
from datetime import datetime
from src.core.config import DISCORD_WEBHOOK_URL

class DiscordNotifier:
    """Sends betting notifications to Discord via Webhook"""
    
    @staticmethod
    def send_notification(title: str, message: str, color: int = 0x00ff00, fields: list = None):
        """
        Send a rich embed notification
        Color: 0x00ff00 (Green/Placed), 0xff0000 (Red/Error), 0xffff00 (Yellow/Info)
        """
        if not DISCORD_WEBHOOK_URL:
            # print("[Discord] No Webhook URL configured.")
            return

        def _send():
            try:
                payload = {
                    "username": "Greyhound Bot",
                    "embeds": [{
                        "title": title,
                        "description": message,
                        "color": color,
                        "timestamp": datetime.utcnow().isoformat(),
                        "footer": {"text": "Auto-Betting System"},
                        "fields": fields if fields else []
                    }]
                }
                
                response = requests.post(
                    DISCORD_WEBHOOK_URL, 
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code not in [200, 204]:
                    print(f"[Discord] Failed to send: {response.status_code} {response.text}")
                    
            except Exception as e:
                print(f"[Discord] Error: {e}")

        # Non-blocking
        threading.Thread(target=_send, daemon=True).start()

    @staticmethod
    def send_bet_placed(bet_details: dict):
        """Format a bet placement notification"""
        # bet_details keys: dog, box, race, track, stake, price, type
        
        # Helper to format money safely
        def fmt_money(val):
            try:
                # If it's a string like "$3.50", strip $ and match float
                if isinstance(val, str):
                    val = val.replace('$', '').replace(',', '')
                f_val = float(val)
                return f"${f_val:.2f}"
            except:
                return str(val)

        fields = [
            {"name": "Stake", "value": fmt_money(bet_details.get('stake', 0)), "inline": True},
            {"name": "Odds", "value": fmt_money(bet_details.get('price', 0)), "inline": True},
            {"name": "Box", "value":str(bet_details.get('box', '?')), "inline": True},
            {"name": "Race", "value": str(bet_details.get('race', '?')), "inline": True},
            {"name": "Track", "value": str(bet_details.get('track', '?')), "inline": True}
        ]
        
        title = f"✅ BET PLACED: {bet_details.get('dog', 'Unknown')}"
        desc = f"Strategy: {bet_details.get('strategy', 'Unknown Strategy')}"
        
        DiscordNotifier.send_notification(title, desc, 0x00ff00, fields)

    @staticmethod
    def send_schedule_summary(bets_list: list):
        """Send a summary of upcoming scheduled bets"""
        if not bets_list:
            return

        # Sort by Time
        sorted_bets = sorted(bets_list, key=lambda x: x.get('time_str', '99:99'))
        
        fields = []
        for bet in sorted_bets:
            # Format: '12:30 | Dog Name (Box 1) | $Stk @ $Odds'
            time = bet.get('time_str', '??:??')
            dog = bet.get('dog', 'Unknown')
            box = bet.get('box', '?')
            track = bet.get('track', '?')[:3].upper() # Short track name
            stake = bet.get('stake', 0)
            price = bet.get('rated_price', 0) # Use Rated or Market? Probably Market implies prediction
            
            # Combine into one line for density
            val = f"Box {box} | {track} | Stake ${stake:.0f}"
            fields.append({
                "name": f"⏰ {time} - {dog}",
                "value": val,
                "inline": False 
            })
            
            # Discord limit is 25 fields per embed. Split if needed?
            # For now, just take top 25
            if len(fields) >= 25:
                fields.append({"name": "...", "value": "More bets scheduled...", "inline": False})
                break
                
        title = f"📅 UPCOMING BETS ({len(bets_list)})"
        desc = "Here is your betting schedule for the upcoming session:"
        
        DiscordNotifier.send_notification(title, desc, 0xffff00, fields)
