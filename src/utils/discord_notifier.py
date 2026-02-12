"""Discord notifications for betting events via webhooks.

Sends rich embed notifications to Discord for bet placements, results, and schedules.
Uses non-blocking threads to avoid delaying main application.

Example:
    >>> from src.utils.discord_notifier import DiscordNotifier
    >>> bet = {'dog': 'Fast Freddy', 'stake': 10, 'price': 6.0, ...}
    >>> DiscordNotifier.send_bet_placed(bet)
    # Sends notification to Discord channel
"""

import requests
import json
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.core.config import DISCORD_WEBHOOK_URL


class DiscordNotifier:
    """Sends betting notifications to Discord via webhook URLs.
    
    Provides static methods for sending various notification types:
    - Bet placement alerts
    - Scheduled bet summaries
    - Custom notifications with rich embeds
    
    All sends are non-blocking (threaded) to avoid application delays.
    
    Example:
        >>> DiscordNotifier.send_notification("Alert", "System started", color=0x00ff00)
    """
    """Sends betting notifications to Discord via Webhook"""
    
    @staticmethod
    def send_notification(
        title: str,
        message: str,
        color: int = 0x00ff00,
        fields: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Send a rich embed notification to Discord.
        
        Args:
            title: Embed title (bold header)
            message: Embed description (main text)
            color: Hex color code for sidebar (default: green 0x00ff00).
                  0x00ff00=Green, 0xff0000=Red, 0xffff00=Yellow
            fields: List of embed fields with 'name', 'value', 'inline' keys (optional)
        
        Example:
            >>> DiscordNotifier.send_notification(
            ...     "Bet Placed",
            ...     "Fast Freddy @ $6.00",
            ...     color=0x00ff00,
            ...     fields=[{'name': 'Stake', 'value': '$10', 'inline': True}]
            ... )
        """
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
    def send_bet_placed(bet_details: Dict[str, Any]) -> None:
        """Send a formatted bet placement notification.
        
        Args:
            bet_details: Dictionary with keys:
                - dog (str): Dog name
                - box (int): Box number
                - race (int): Race number
                - track (str): Track name
                - stake (float): Bet amount
                - price (float): Odds
                - type/strategy (str): Strategy used
        
        Example:
            >>> bet = {
            ...     'dog': 'Fast Freddy',
            ...     'box': 1,
            ...     'race': 5,
            ...     'track': 'Wentworth Park',
            ...     'stake': 10.0,
            ...     'price': 6.0,
            ...     'strategy': 'V44_BACK'
            ... }
            >>> DiscordNotifier.send_bet_placed(bet)
        """
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
    def send_schedule_summary(bets_list: List[Dict[str, Any]]) -> None:
        """Send a summary of upcoming scheduled bets.
        
        Formats multiple bets into a single notification with time-sorted list.
        Limits to 25 bets (Discord embed field limit).
        
        Args:
            bets_list: List of bet dictionaries with keys:
                - time_str (str): Race time (HH:MM format)
                - dog (str): Dog name
                - box (int): Box number
                - track (str): Track name
                - stake (float): Bet amount
                - rated_price (float): Predicted odds
        
        Example:
            >>> bets = [
            ...     {'time_str': '14:30', 'dog': 'Dog A', 'box': 1, ...},
            ...     {'time_str': '15:00', 'dog': 'Dog B', 'box': 3, ...}
            ... ]
            >>> DiscordNotifier.send_schedule_summary(bets)
        """
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
